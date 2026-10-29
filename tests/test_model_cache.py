from types import SimpleNamespace
from unittest.mock import patch

from lm_eval.api.model import CachingLM


class DummyLM:
    def __init__(self):
        self.generate_calls = 0

    def set_cache_hook(self, cache_hook):
        self.cache_hook = cache_hook

    def generate_until(self, requests):
        self.generate_calls += 1
        return ["generated"] * len(requests)


def test_caching_lm_disables_sqlitedict_outer_stack(tmp_path):
    cache_path = tmp_path / "responses.db"

    with patch("sqlitedict.SqliteDict") as sqlite_dict:
        CachingLM(DummyLM(), str(cache_path))

    sqlite_dict.assert_called_once_with(
        str(cache_path), autocommit=True, outer_stack=False
    )


def test_caching_lm_can_resume_sampled_generations_when_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LMEVAL_CACHE_SAMPLED_GENERATIONS", "1")
    model = DummyLM()
    cached_model = CachingLM(model, str(tmp_path / "responses.db"))
    request = SimpleNamespace(args=("prompt", {"do_sample": True}))

    assert cached_model.generate_until([request]) == ["generated"]
    assert cached_model.generate_until([request]) == ["generated"]
    assert model.generate_calls == 1

    cached_model.dbdict.close()
