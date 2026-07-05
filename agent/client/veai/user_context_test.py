import unittest
from importlib.resources import files

from agent.client.veai import _get_context

TEST_RESOURCES = "test_resources"


class UserContextCase(unittest.TestCase):
    def test_parse(self):
        veai_project_information_file = files(__package__).joinpath(TEST_RESOURCES, "veai_project_information.txt")
        veai_project_information = veai_project_information_file.read_text(encoding="utf-8")
        context = _get_context(veai_project_information)
        self.assertIsNotNone("", context)
        self.assertEqual('Windows 11, version: 10.0, arch: amd64', context.os)
        self.assertEqual('C:\\project', context.workdir)


if __name__ == '__main__':
    unittest.main()
