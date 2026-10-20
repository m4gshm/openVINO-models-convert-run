"""Tests of the scheduler configuration built out of the server options."""
import unittest

from agent.openai import SchedulerOpts, get_default_scheduler_opts
from agent.openvino.scheduler import build_scheduler_config


class SchedulerConfigCase(unittest.TestCase):
    """Tests of the translation of the scheduler options."""

    def test_default_options_enable_the_prefix_caching(self):
        """The built-in defaults turn the prefix caching on."""
        config = build_scheduler_config(get_default_scheduler_opts(), 4096)
        self.assertTrue(config.enable_prefix_caching)

    def test_cache_size_is_applied(self):
        """The cache size of the options reaches the config."""
        config = build_scheduler_config(SchedulerOpts(cache_size=2048), 4096)
        self.assertEqual(2048, config.cache_size)

    def test_max_num_seqs_is_applied(self):
        """The sequence limit of the options reaches the config."""
        config = build_scheduler_config(SchedulerOpts(max_num_seqs=8), 4096)
        self.assertEqual(8, config.max_num_seqs)

    def test_dynamic_split_fuse_sets_the_batched_token_limit(self):
        """A dynamic split fuse needs a batched token limit of its own."""
        config = build_scheduler_config(SchedulerOpts(dynamic_split_fuse=True,
                                                     max_num_batched_tokens=4096), 8192)
        self.assertTrue(config.dynamic_split_fuse)
        self.assertEqual(4096, config.max_num_batched_tokens)

    def test_disabled_dynamic_split_fuse_bounds_the_batch_by_the_prompt(self):
        """Without a dynamic split fuse the batch does not exceed the prompt length."""
        config = build_scheduler_config(SchedulerOpts(dynamic_split_fuse=False), 8192)
        self.assertFalse(config.dynamic_split_fuse)
        self.assertEqual(8192, config.max_num_batched_tokens)

    def test_cache_eviction_stays_off(self):
        """The cache eviction is not enabled by the options."""
        config = build_scheduler_config(SchedulerOpts(), 4096)
        self.assertFalse(config.use_cache_eviction)

    def test_sparse_attention_is_applied(self):
        """The sparse attention parameters reach the config."""
        config = build_scheduler_config(SchedulerOpts(use_sparse_attention=True), 4096)
        self.assertTrue(config.use_sparse_attention)
        self.assertIsNotNone(config.sparse_attention_config)


if __name__ == '__main__':
    unittest.main()
