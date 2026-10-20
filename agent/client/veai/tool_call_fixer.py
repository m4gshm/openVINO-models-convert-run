"""Repair of model-generated tool calls: per-tool argument sanitizing for Veai IDE tools."""
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable

import json_repair

from agent.client.user_context import UserContext, ROOT, DEFAULT_DEPTH
from agent.client.veai.tool import edit_file, read_file, write_file, search_for_text, ask_user_with_options, list_dir, \
    search_file_by_name, file_structure, run_command, run_configuration, safe_delete
from agent.client.veai.tool.ask_user_with_options import AskUserWithOptions
from agent.client.veai.tool.edit_file import EditFile
from agent.client.veai.tool.file_structure import FileStructure
from agent.client.veai.tool.list_dir import ListDir
from agent.client.veai.tool.read_file import ReadFile
from agent.client.veai.tool.run_command import RunCommand
from agent.client.veai.tool.run_configuration import RunConfiguration
from agent.client.veai.tool.safe_delete import SafeDelete
from agent.client.veai.tool.search_file_by_name import SearchFileByName
from agent.client.veai.tool.search_for_text import SearchForText
from agent.client.veai.tool.write_file import WriteFile
from agent.client.veai.tool_call_args import as_bool_or_none, as_int_or_none, clean_file_path, \
    find_target_file_from_prev_tool_call, fix_windows_path, get_args, get_one_of, get_target_file
from agent.openai.chat_completions_api import ChatCompletionFunctionToolParam, FunctionDefinitionParameters
from agent.parser import ParsedFunctionCall
from agent.parser.json_fixer import try_to_parse_json_arguments

NEW_LINE_UNIX = "\n"
NEW_LINE_WINDOWS = "\r\n"

GEMMA_4 = "Gemma4ForConditionalGeneration"

log = logging.getLogger(__name__)


ToolFixer = Callable[[ParsedFunctionCall, UserContext | None], list[ParsedFunctionCall] | ParsedFunctionCall]


def veai_fix_incorrect_arguments(function: ParsedFunctionCall, user_context: UserContext | None = None) -> \
        list[ParsedFunctionCall] | ParsedFunctionCall:
    """Dispatch the parsed tool call to the per-tool argument fixer."""
    fixer = _TOOL_FIXERS.get(function.name)
    if fixer:
        return fixer(function, user_context)
    return function


def fix_ask_user_with_options(function: ParsedFunctionCall, _context: UserContext | None) -> ParsedFunctionCall:
    """Repair ask_user_with_options arguments: require options, question and choice mode."""
    invalid = False
    args = get_args(function)
    options_raw = args.get("options")

    is_multiple_choice = as_bool_or_none(args.get("is_multiple_choice"), "is_multiple_choice")
    if not is_multiple_choice:
        invalid = True
        is_multiple_choice = False
    question = args.get("question")
    if not question:
        invalid = True
        question = "[*]" if is_multiple_choice else "(*)"
    options: Any = None
    if options_raw:
        if isinstance(options_raw, str):
            invalid = True
            try:
                options = json.loads(options_raw)
            except json.decoder.JSONDecodeError as e:
                log.error("bad options of function '%s', options: '%s': %s", function.name, options_raw, e)
                options = json_repair.loads(options_raw)
                log.info("repaired options '%s'", options)
        elif isinstance(options_raw, list):
            options = options_raw
        else:
            log.error("unexpected options type, function '%s', args '%s', options type %s",
                      function.name, args, type(options_raw))
    else:
        log.error("missing options in args, function '%s', args '%s'", function.name, args)

    if invalid:
        log.info("function after repairing: %s, question='%s', is_multiple_choice=%s, options=%s",
                 function.name, question, is_multiple_choice, options)
        return AskUserWithOptions().new_call(question, options, is_multiple_choice)
    return function


def fix_file_structure(function: ParsedFunctionCall, context: UserContext | None) -> ParsedFunctionCall:
    """Repair file_structure arguments: resolve and validate the target file."""
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)
    target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    if found:
        invalid = True

    if not target_file:
        log.error("tool call error: tool=%s, target_file is empty but required", function.name)

    if invalid:
        new_function = FileStructure().new_call(target_file)
        return new_function
    return function


def fix_edit_file(function: ParsedFunctionCall, context: UserContext | None = None) -> \
        list[ParsedFunctionCall] | ParsedFunctionCall:
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks  # model-output repair for the richest IDE tool; splitting is a separate refactor
    """Repair edit_file arguments: normalize edits, resolve target file, rebuild old_text."""
    args = get_args(function)
    target_file: Any = None
    target_file, _invalid = get_target_file(args, context)
    edits: Any = args.get("edits")

    if isinstance(edits, str):
        edits = try_to_parse_json_arguments(edits)

    if edits:
        edits, unused_anonymous_edits = handle_edits(edits)
    else:
        edits = [{}]
        unused_anonymous_edits = []

    first_edit = edits[0]

    # recheck edits fullness:
    global_new_text = get_one_of(args, ["new_text", "new_content", "content"])
    if global_new_text:
        if not "new_text" in first_edit:
            first_edit["new_text"] = global_new_text
        else:
            is_set_new_text = False
            for edit in edits:
                if "new_text" not in edit and "old_text" in edit:
                    is_set_new_text = True
                    edit["new_text"] = global_new_text
            if is_set_new_text:
                del args["new_text"]

    global_old_text = args.get("old_text")
    if global_old_text:
        if not "old_text" in first_edit:
            first_edit["old_text"] = global_old_text
        else:
            is_set_old_text = False
            for edit in edits:
                if "old_text" not in edit and "new_text" in edit:
                    is_set_old_text = True
                    edit["old_text"] = global_old_text
            if is_set_old_text:
                del args["old_text"]

    if not target_file and edits:
        # gemma 4
        # try to search in edits
        for i, edit in enumerate(edits):
            if isinstance(edit, dict):
                target_file = edit.get("target_file")
                if target_file:
                    del edit["target_file"]
                    if len(edit) == 0:
                        del edits[i]
                    break
    if not target_file and unused_anonymous_edits:
        expect_target_file = False
        for v in unused_anonymous_edits:
            if expect_target_file:
                target_file = v
                break
            if "target_file" == v:
                expect_target_file = True
            elif "target_file" in v:
                pass
    if not target_file:
        target_file = args.get("file_path")
        if target_file:
            del args["file_path"]

    if target_file or edits:
        allow_multiple_matches = as_bool_or_none(args.get("allow_multiple_matches"), "allow_multiple_matches")
        if allow_multiple_matches is None:
            allow_multiple_matches = as_bool_or_none(args.get("allowed_multiple_matches"), "allowed_multiple_matches")
            if not allow_multiple_matches is None:
                _invalid = True
                del args["allowed_multiple_matches"]

        if allow_multiple_matches is None:
            _invalid = True
            allow_multiple_matches = False

        if isinstance(edits, list):
            if len(edits) > 1:
                # gemma 4
                # try to merge
                _set_prev = False
                _merged = False
                prev_old_text: str | None = None
                prev_new_text: str | None = None
                for edit in edits:
                    if isinstance(edit, dict):
                        old_text_lies = edit.get("old_text")
                        new_text_lines = edit.get("new_text")
                        if prev_old_text is None and prev_new_text is None:
                            _set_prev = True
                            prev_old_text = old_text_lies if isinstance(old_text_lies, str) else None
                            prev_new_text = new_text_lines if isinstance(new_text_lines, str) else None
                        else:
                            if prev_old_text and isinstance(old_text_lies, str) and isinstance(
                                    prev_new_text, str):
                                in_prev = old_text_lies.startswith(prev_old_text)
                                if in_prev:
                                    log.debug("merge in pre, new_text '%s' with '%s' for old_text '%s' and '%s'",
                                              prev_new_text, new_text_lines, prev_old_text, old_text_lies)
                                    prev_new_text = prev_new_text + (new_text_lines or "")
                                    prev_old_text = old_text_lies
                                    _merged = True
                    elif isinstance(edit, list):
                        log.error("unexpected edit type in edits: type=%s edit=%s, edits=%s", type(edit), edit, edits)
                    else:
                        log.error("unexpected edit type in edits: type=%s edit=%s, edits=%s", type(edit), edit, edits)

                if prev_old_text and prev_new_text:
                    edits = [{"new_text": prev_new_text, "old_text": prev_old_text}]
                    log.debug("merged edits=%s", edits)
                    _invalid = True

        target_file, _ = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
        target_file = clean_file_path(target_file)
        if not target_file:
            log.error("tool call error: tool=%s, target_file is empty but required", function.name)
        else:
            for edit in edits:
                if "old_text" not in edit:
                    new_text = edit.get("new_text", None)
                    if new_text:
                        old_text, new_text_rebuild = rebuild_edit_texts_using_file_content(new_text, target_file,
                                                                                           context)
                        if not old_text:
                            log.debug("cannot reconstruct old_text from file content, target_file='%s'", target_file)
                        else:
                            log.debug("edits reconstructed from file content, "
                                      "target_file='%s':\nold_text:\n%snew_text:\n%s",
                                      target_file, old_text, new_text)
                            edit["new_text"] = new_text_rebuild
                            edit["old_text"] = old_text

        return EditFile().new_call(target_file, edits, allow_multiple_matches=allow_multiple_matches)
    return function


def rebuild_edit_texts_using_file_content(new_text: str, target_file: str, context: UserContext | None) -> tuple[
    str | None, str]:
    # pylint: disable=too-many-locals,too-many-branches  # alignment search over file lines; splitting is a separate refactor
    """Reconstruct old_text by locating new_text inside the tracked file content."""
    file_content_bytes: bytes | None = context.files.get_file_content(target_file) if context else None
    file_content_str: str | None = file_content_bytes.decode("utf-8") if file_content_bytes else None
    file_content_lines: list[str] = split_lines(file_content_str) if file_content_str else []

    if not file_content_lines:
        context_files = context.files.get_files() if context else []
        log.debug("no target_file content in context, target_file=%s, context_files=%s",
                  target_file, context_files)
        return None, ""

    stop = False
    new_text_lines = split_lines(new_text)
    new_text_merge_point_start = 0
    old_text_merge_point_start = -1
    old_text_merge_point_end = -1
    while not stop and new_text_merge_point_start < len(new_text_lines):
        first_new_text_line = new_text_lines[new_text_merge_point_start]
        i = old_text_merge_point_end + 1
        while i < len(file_content_lines):
            old_line = file_content_lines[i]
            if old_line == first_new_text_line:
                if old_text_merge_point_start == -1:
                    old_text_merge_point_start = i
                old_text_merge_point_end = i
                break
            if old_text_merge_point_start != -1:
                # new_text_merge_point_start -= 1
                stop = True
                break
            i += 1
        if not stop:
            new_text_merge_point_start += 1

    # start is found
    old_text_lines = file_content_lines[
        old_text_merge_point_start: old_text_merge_point_end + 1] if old_text_merge_point_start != -1 else None

    new_text_merge_point_end = len(new_text_lines) - 1
    if old_text_merge_point_start != -1 and len(new_text_lines) > 2:
        old_text_point = len(file_content_lines) - 1
        i = new_text_merge_point_end
        stop = False
        while old_text_point > old_text_merge_point_end and not stop and i > new_text_merge_point_start:
            last_new_text_line = new_text_lines[i]
            while old_text_point > old_text_merge_point_end:
                old_line = file_content_lines[old_text_point]
                if old_line == last_new_text_line:
                    # go up
                    old_text_point -= 1
                    new_text_merge_point_end = i - 1
                    break
                if i < new_text_merge_point_end:
                    stop = True
                    break
                old_text_point -= 1
            if not stop:
                i -= 1

    new_text_lines = new_text_lines[new_text_merge_point_start: new_text_merge_point_end + 1]
    old_text_join = "\n".join(old_text_lines) if old_text_lines else None
    new_text_join = "\n".join(new_text_lines)
    return old_text_join, (old_text_join if old_text_join else "") + new_text_join


def split_lines(file_content_str: str) -> list[str]:
    """Split file content by the dominant newline style (Windows or Unix)."""
    return file_content_str.split(NEW_LINE_WINDOWS if NEW_LINE_WINDOWS in file_content_str else NEW_LINE_UNIX)


def try_find_target_file_from_prev_tool_call_if_need(args: dict[str, Any], context: UserContext | None,
                                                     function: ParsedFunctionCall,
                                                     target_file: str | Any) -> tuple[Any, bool]:
    """For Gemma4 recover a missing target_file from the previous tool call.

    Returns the resolved target file and whether it was recovered.
    """
    if not target_file and (context and GEMMA_4 in context.model_architectures):
        prev_target_file = find_target_file_from_prev_tool_call(args, context, function.name, target_file)
        return prev_target_file, not prev_target_file is None
    return target_file, False


class _EditNormalizer:  # pylint: disable=too-many-instance-attributes,too-few-public-methods  # state of the edits normalization
    """Normalizes the malformed edits payloads of the model.

    The model may send the edits as a single dict, as a list of dicts or as a
    flat list of values; the normalizer turns them into a list of edit dicts and
    collects the values that could not be attributed to any edit.
    """

    def __init__(self, edits: Any):
        self.items = _as_edit_list(edits)
        self.new_text: Any = None
        self.new_text_i: int | None = None
        self.old_text: Any = None
        self.old_text_i: int | None = None
        self.on_delete_i: list[int] = []
        self.anonymous_edits: list[Any] = []

    def normalize(self) -> tuple[list[Any], list[Any]]:
        """Normalize the edits into a list of edit dicts plus the unattributed values."""
        i = 0
        while i < len(self.items):
            edit = self.items[i]
            if isinstance(edit, dict):
                self._handle_dict_edit(edit, i)
            elif isinstance(edit, (list, set)):
                self._handle_group_edit(edit, i)
            elif edit in ("old_text", "new_text"):
                i = self._handle_keyword_edit(edit, i)
            else:
                self.anonymous_edits.append(edit)
                self.on_delete_i.append(i)
            i += 1

        _delete_indices(self.items, self.on_delete_i)

        unused_anonymous_edits: list[Any] = []
        if self.anonymous_edits:
            dirty_edits, unused_anonymous_edits = _group_anonymous_edits(self.anonymous_edits)
            self.items.extend(dirty_edits)
            unused_anonymous_edits.extend(handle_edits(self.items)[1])

        _merge_incomplete_edits(self.items)

        return self.items, unused_anonymous_edits

    def _handle_dict_edit(self, edit: dict[str, Any], i: int) -> None:
        """Take the texts of a well formed edit, or merge it with its neighbour."""
        if len(edit) == 0:
            self.on_delete_i.append(i)
            return
        edit_new_text = replace_by_possibles("new_text", ["new_config", "new_text="], edit)
        edit_old_text = replace_by_possibles("old_text", ["old_config", "old_text="], edit)

        if not self.new_text_i and edit_new_text:
            self.new_text = edit_new_text
            self.new_text_i = i
        if not self.old_text_i and edit_old_text:
            self.old_text = edit_old_text
            self.old_text_i = i

        if self.old_text_i == self.new_text_i:
            # the edit carries both texts, as expected
            self.old_text_i = None
            self.new_text_i = None
        elif self.old_text_i is not None and self.new_text_i is not None:
            # the texts landed in two different edits: merge them back
            self.items[self.new_text_i]["old_text"] = self.old_text
            del self.items[self.old_text_i]["old_text"]
            if len(self.items[self.old_text_i]) == 0:
                self.on_delete_i.append(self.old_text_i)
            self.old_text_i = None
            self.new_text_i = None

    def _handle_group_edit(self, edit: list[Any] | set[Any], i: int) -> None:
        """Take a group of loose values as the text the pending edit is missing."""
        first = next(iter(edit)) if len(edit) > 0 else None
        if self.new_text is None and self.old_text and self.old_text_i is not None:
            self.new_text = first
            self.items[self.old_text_i]["new_text"] = first
            _drop_first(edit)
        elif self.old_text is None and self.new_text and self.new_text_i is not None:
            self.old_text = first
            self.items[self.new_text_i]["old_text"] = first
            _drop_first(edit)
        else:
            self.anonymous_edits.extend(edit)
        self.on_delete_i.append(i)

    def _handle_keyword_edit(self, edit: Any, i: int) -> int:
        """Pair a text keyword with the value that follows it."""
        next_i = i + 1
        next_e = self.items[next_i] if next_i < len(self.items) else None
        if not next_e is None:
            self.items[i] = {edit: next_e}
            self.on_delete_i.append(next_i)
            return next_i
        self.anonymous_edits.append(edit)
        self.on_delete_i.append(i)
        return i


def handle_edits(edits: Any) -> tuple[list[Any], list[Any]]:
    """Normalize a malformed edits payload into a list of edit dicts."""
    return _EditNormalizer(edits).normalize()


def _as_edit_list(edits: Any) -> list[Any]:
    """Normalize the edits payload into a mutable list of edit items."""
    if isinstance(edits, dict):
        return [edits]
    if isinstance(edits, Iterable) and not isinstance(edits, (str, bytes)):
        return list(edits)
    log.warning("unexpected edits type='%s', edits='%s'", type(edits), edits)
    return []


def _drop_first(edit: list[Any] | set[Any]) -> None:
    """Drop the first value of the edit item; a set has no order, so it is left as it is."""
    if isinstance(edit, list) and edit:
        del edit[0]


def _delete_indices(items: list[Any], indices: list[int]) -> None:
    """Remove the items at the given positions, the last one first."""
    for i in reversed(indices):
        try:
            del items[i]
        except IndexError as e:
            log.error("error on del edits[%s]: %s, edits=%s", i, e, items)
            raise e


def _group_anonymous_edits(anonymous_edits: list[Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Group the loose values into edit dicts, keeping the values that fit none."""
    expect_next_next = False
    expect_old_text = False
    dirty_edits: list[dict[str, Any]] = []
    unused: list[Any] = []
    edit: dict[str, Any] = {}
    for v in anonymous_edits:
        if expect_next_next:
            edit["new_text"] = v
            expect_next_next = False
        elif expect_old_text:
            edit["old_text"] = v
            expect_old_text = False
        elif v == "new_text":
            if "new_text" in edit:
                dirty_edits.append(edit)
                edit = {}
            expect_next_next = True
        elif v == "old_text":
            if "old_text" in edit:
                dirty_edits.append(edit)
                edit = {}
            expect_old_text = True
        else:
            unused.append(v)

    if edit:
        dirty_edits.append(edit)
    return dirty_edits, unused


def _merge_incomplete_edits(items: list[Any]) -> None:
    """Merge an edit that carries only one text with the edit next to it."""
    on_delete_i: list[int] = []
    i = 0
    while i < len(items):
        edit = items[i]
        if isinstance(edit, dict):
            merged_i = _merge_with_next_edit(items, edit, i)
            if merged_i is not None:
                on_delete_i.append(merged_i)
                i = merged_i
        i += 1

    _delete_indices(items, on_delete_i)


def _merge_with_next_edit(items: list[Any], edit: dict[str, Any], i: int) -> int | None:
    """Complete an edit that carries only one text from the edit next to it.

    Returns the position of the edit that was merged in and must be dropped.
    """
    if "new_text" in edit and "old_text" in edit:
        return None
    next_i = i + 1
    next_edit = items[next_i] if next_i < len(items) else None
    if not isinstance(next_edit, dict):
        return None
    if "new_text" in next_edit and "old_text" in next_edit:
        return None
    if "new_text" in edit and "old_text" in next_edit:
        edit["old_text"] = next_edit["old_text"]
        return next_i
    if "old_text" in edit and "new_text" in next_edit:
        edit["new_text"] = next_edit["new_text"]
        return next_i
    return None


def replace_by_possibles(expected_key: str, possible_keys: list[str], params: dict[str, Any]) -> Any:
    """Return value of expected_key, trying alternative keys and moving them in place."""
    edit_new_text = params.get(expected_key)
    if edit_new_text is None:
        for key in possible_keys:
            replaced = find_and_replace(key, params)
            if not replaced is None:
                return replaced
    return edit_new_text


def find_and_replace(key: str, edit: dict[str, Any]) -> Any | None:
    """Move the value stored under key to the new_text slot, deleting the old key."""
    edit_new_text = edit.get(key)
    if edit_new_text:
        edit["new_text"] = edit_new_text
        del edit[key]
    return edit_new_text


def fix_write_file(function: ParsedFunctionCall, context: UserContext | None = None) -> ParsedFunctionCall:
    """Repair write_file arguments: require target_file and content, default allow_overwrite."""
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)
    content = args.get("content")

    # target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    # if found:
    #     invalid |= found

    if target_file and content:
        allow_overwrite = args.get("allow_overwrite")

        if not allow_overwrite:
            invalid = True
            allow_overwrite = True

        if invalid:
            new_function = WriteFile().new_call(target_file, content, allow_overwrite=allow_overwrite)
            return new_function
    else:
        log.error("no required args for function %s, args=%s, required args = ['target_file', 'content']",
                  function.name, args)

    return function


def fix_safe_delete(function: ParsedFunctionCall, _context: UserContext | None) -> ParsedFunctionCall:
    """Repair safe_delete arguments: rename file_path keys to path."""
    args = get_args(function)
    targets: list[dict[str, Any]] = args.get("targets") or []
    invalid = False
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and not "path" in target:
                path = target.get("file_path", None)
                if path:
                    target["path"] = path
                    del target["file_path"]
                    invalid = True

    if invalid:
        force = args.get("force", False)
        dry_run = args.get("dry_run", False)
        new_function = SafeDelete().new_call(targets=targets, force=force, dry_run=dry_run)
        return new_function

    return function


def fix_search_for_text(function: ParsedFunctionCall, context: UserContext | None) -> ParsedFunctionCall:
    """Repair search_for_text arguments: resolve target and snippet, default case sensitivity."""
    args = get_args(function)
    target_path_or_url = args.get("target_path_or_url")
    if not target_path_or_url:
        target_path_or_url = get_one_of(args, ["target_path", "target_path_url", "search_directory"])

    text_snippet = args.get("text_snippet")
    if target_path_or_url and text_snippet:
        target_path_or_url, _fixed = fix_windows_path(target_path_or_url, context)
        is_case_sensitive = as_bool_or_none(args.get("is_case_sensitive"), "is_case_sensitive") or False
        new_function = SearchForText().new_call(target_path_or_url, text_snippet, is_case_sensitive)
        return new_function

    return function


def fix_search_file_by_name(function: ParsedFunctionCall, context: UserContext | None = None) -> ParsedFunctionCall:
    """Repair search_file_by_name arguments: recover pattern and directory from alternatives."""
    args = get_args(function)
    glob_pattern = args.get("glob_pattern")
    invalid = not glob_pattern
    if invalid:
        # gemma 4
        glob_pattern = args.get("glob")

    invalid = not glob_pattern
    if invalid:
        # gemma 4
        glob_pattern = args.get("pattern")

    invalid = not glob_pattern
    if invalid:
        # gemma 4
        glob_pattern = args.get("query")

    search_directory = args.get("search_directory")
    if not search_directory:
        invalid = True
        # try to find by partial equality
        target_path_or_url = args.get("target_path_or_url")
        if not target_path_or_url is None:
            path = Path(target_path_or_url)
            if glob_pattern is None:
                glob_pattern = str(path.name)
            search_directory = str(path.parent)
        else:
            search_directory = ROOT
        invalid = True

    search_directory, fixed = fix_windows_path(search_directory, context)
    if fixed:
        invalid = True

    if invalid:
        log.info("fix invalid %s: glob_pattern=%s, search_directory=%s",
                 function.name, glob_pattern, search_directory)
        new_function = SearchFileByName().new_call(glob_pattern, search_directory)
        return new_function
    return function


def fix_read_file(function: ParsedFunctionCall, context: UserContext | None = None) -> ParsedFunctionCall:
    """Repair read_file arguments: require target_file, sanitize line range and offset."""
    args = get_args(function)
    target_file, invalid = get_target_file(args, context, ["content"])

    anonymous_arguments = function.anonymous_arguments
    if not target_file and anonymous_arguments:
        invalid = True
        target_file = anonymous_arguments[0]

    target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    if found:
        invalid = True

    if not target_file:
        log.error("tool call error: tool=%s, target_file is empty but required", function.name)
    else:
        start_line, fixed = as_int_or_none(args.get("start_line"), "start_line")
        invalid |= fixed
        end_line, fixed = as_int_or_none(args.get("end_line"), "end_line")
        invalid |= fixed
        line_offset, fixed = as_int_or_none(args.get("line_offset"), "line_offset")
        invalid |= fixed

        if line_offset and (start_line or end_line):
            invalid = True
            line_offset = None

        if not line_offset:
            if not start_line:
                invalid = True
                start_line = 1
            if not end_line:
                invalid = True
                end_line = 500
        if invalid:
            log.info("fix invalid %s: target_file=%s, start_line=%s, end_line=%s, line_offset=%s",
                     function.name, target_file, start_line, end_line, line_offset)
            new_function = ReadFile().new_call(target_file=target_file, start_line=start_line, end_line=end_line,
                                               line_offset=line_offset)
            return new_function

    return function


def fix_list_dir(function: ParsedFunctionCall, context: UserContext | None) -> ParsedFunctionCall:
    """Repair list_dir arguments: default directory to root and clamp the depth."""
    args = get_args(function)
    directory_path = args.get("directory_path")
    invalid = False
    if not directory_path:
        invalid = True
        # gemma4 case
        directory_path = args.get("dir")

    if not directory_path:
        invalid = True
        directory_path = ROOT

    depth = args.get("depth")
    if not depth:
        invalid = True
        # the depth limit depends on the directory the listing starts from
        depth = context.max_list_dir_depth(directory_path) if context else None
        if depth is None:
            depth = DEFAULT_DEPTH

    directory_path, fixed = fix_windows_path(directory_path, context)
    if fixed:
        invalid = True

    if invalid:
        log.info("fix invalid %s: directory_path=%s, depth=%s", function.name, directory_path, depth)
        new_function = ListDir().new_call(directory_path=directory_path, depth=depth)
        return new_function
    return function


def fix_run_command(function: ParsedFunctionCall, context: UserContext | None) -> ParsedFunctionCall:
    """Repair run_command arguments: default safe flags and working directory."""
    args = get_args(function)

    invalid = False

    command = args.get("command")
    is_background = args.get("is_background")
    if not is_background:
        invalid = True
        is_background = False
    safe_to_run = args.get("safe_to_run")
    if not safe_to_run:
        invalid = True
        safe_to_run = False

    working_directory = args.get("working_directory")
    if working_directory is None:
        working_directory = ROOT
        invalid = True

    working_directory, fixed = fix_windows_path(working_directory, context)
    if fixed:
        invalid = True

    if invalid:
        log.info("fix invalid %s: command=%s, working_directory=%s, is_background=%s, safe_to_run=%s",
                 function.name, command, working_directory, is_background, safe_to_run)
        new_function = RunCommand().new_call(command=command, working_directory=working_directory,
                                             is_background=is_background, safe_to_run=safe_to_run)
        return new_function
    return function


def veai_fix_tool_definition_optional_property_as_null_type(
        tool: ChatCompletionFunctionToolParam) -> ChatCompletionFunctionToolParam:
    """Replace optional-property union types with the base type and unrequire them."""
    function = tool.function
    function.parameters = _fix_tool_definition_optional_property_as_null_type(function.parameters, function.name)
    return tool


def _fix_tool_definition_optional_property_as_null_type(parameters: FunctionDefinitionParameters,
                                                        parent_name: str) -> FunctionDefinitionParameters:
    properties = parameters.properties
    required = parameters.required
    for prop_name, prop_params in properties.items():
        params: dict[str, Any] = prop_params
        prop_type = params.get("type")
        if isinstance(prop_type, list):
            if len(prop_type) >= 1:
                opt = False
                for i in range(1, len(prop_type)):
                    if prop_type[i] == "null":
                        opt = True
                        break

                new_type = prop_type[0]
                params["type"] = new_type

                if opt and required:
                    required.remove(prop_name)
                    parameters.required = required

                log.debug("fix parameter type: parent object '%s', property '%s', "
                          "new type '%s', old type '%s', optional %s",
                          parent_name, prop_name, new_type, prop_type, opt)
                prop_type = new_type
                if prop_type == "object":
                    parsed_params = FunctionDefinitionParameters.model_validate(params)
                    fixed_params = _fix_tool_definition_optional_property_as_null_type(parsed_params, prop_name)
                    properties[prop_name] = fixed_params.model_dump()

    return parameters


def fix_run_configuration(function: ParsedFunctionCall, context: UserContext | None = None) -> ParsedFunctionCall:
    """Repair run_configuration arguments: require target file and config name."""
    args = get_args(function)
    target_file, invalid = get_target_file(args, context)
    configuration_name = args.get("configuration_name")

    target_file, found = try_find_target_file_from_prev_tool_call_if_need(args, context, function, target_file)
    if found:
        invalid |= found

    if target_file and configuration_name:
        line_number, fixed = as_int_or_none(args.get("line_number"), "line_number")
        invalid |= fixed
        if line_number is None:
            line_number = 0
            invalid = True

        timeout, fixed = as_int_or_none(args.get("timeout"), "timeout")
        invalid |= fixed

        if invalid:
            configuration_run_arguments = args.get("configuration_run_arguments")
            configuration_environment_variables = args.get("configuration_environment_variables")
            files_to_collect_coverage = args.get("files_to_collect_coverage")
            new_function = RunConfiguration().new_call(
                target_file=target_file, configuration_name=configuration_name,
                line_number=line_number, timeout=timeout,
                configuration_run_arguments=configuration_run_arguments,
                configuration_environment_variables=configuration_environment_variables,
                files_to_collect_coverage=files_to_collect_coverage)
            return new_function
    else:
        log.error("no required args for function %s, args=%s, required args = ['target_file', 'configuration_name']",
                  function.name, args)
    return function


_TOOL_FIXERS: dict[str, ToolFixer] = {
    run_command.FUNCTION_NAME: fix_run_command,
    list_dir.FUNCTION_NAME: fix_list_dir,
    file_structure.FUNCTION_NAME: fix_file_structure,
    edit_file.FUNCTION_NAME: fix_edit_file,
    write_file.FUNCTION_NAME: fix_write_file,
    read_file.FUNCTION_NAME: fix_read_file,
    search_for_text.FUNCTION_NAME: fix_search_for_text,
    search_file_by_name.FUNCTION_NAME: fix_search_file_by_name,
    ask_user_with_options.FUNCTION_NAME: fix_ask_user_with_options,
    run_configuration.FUNCTION_NAME: fix_run_configuration,
    safe_delete.FUNCTION_NAME: fix_safe_delete,
}
