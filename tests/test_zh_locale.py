"""Tests for the fork's locale-aware Chinese support (option B).

Home Assistant language drives the behaviour: Chinese HA -> Chinese
prompts / translations / timeline strings, anything else -> English.
A prompt the user typed themselves always wins over both.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from custom_components.llmvision.const import (
    DATA_EXTRACTION_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TITLE_PROMPT,
    DEFAULT_SYSTEM_PROMPT_ZH,
    DEFAULT_TITLE_PROMPT_ZH,
    DATA_EXTRACTION_PROMPT_ZH,
    DEFAULT_SUMMARY_PROMPT_ZH,
    get_data_extraction_prompt,
    get_default_summary_prompt,
    get_default_system_prompt,
    get_default_title_prompt,
    get_hass_language,
    get_memory_reference_prompt,
    is_chinese_language,
)
from custom_components.llmvision.providers import Provider

ROOT = Path(__file__).resolve().parent.parent / "custom_components" / "llmvision"


class TestIsChineseLanguage:
    def test_ha_language_codes(self):
        for lang in ("zh-Hans", "zh-Hant", "zh-CN", "zh-TW", "zh"):
            assert is_chinese_language(lang) is True

    def test_legacy_and_label(self):
        assert is_chinese_language("cn") is True
        assert is_chinese_language("Chinese") is True

    def test_other_languages(self):
        for lang in ("en", "en-US", "de", "ja", "fr", "", None):
            assert is_chinese_language(lang) is False

    def test_non_string_is_not_chinese(self):
        # conftest's mock_hass leaves hass.config.language as a Mock;
        # that must never be treated as Chinese.
        assert is_chinese_language(object()) is False

    def test_get_hass_language(self):
        assert get_hass_language(SimpleNamespace(config=SimpleNamespace(language="zh-Hans"))) == "zh-Hans"
        assert get_hass_language(SimpleNamespace(config=SimpleNamespace(language=object()))) is None
        assert get_hass_language(SimpleNamespace()) is None
        assert get_hass_language(object()) is None


class TestLocalePrompts:
    def test_chinese_prompts_are_chinese(self):
        for prompt in (
            DEFAULT_SYSTEM_PROMPT_ZH,
            DEFAULT_TITLE_PROMPT_ZH,
            DATA_EXTRACTION_PROMPT_ZH,
            DEFAULT_SUMMARY_PROMPT_ZH,
        ):
            assert any("\u4e00" <= ch <= "\u9fff" for ch in prompt)

    def test_english_defaults_unchanged(self):
        assert "Analyze the images" in DEFAULT_SYSTEM_PROMPT
        assert "Generate a clear event title" in DEFAULT_TITLE_PROMPT
        assert "{data_format}" in DATA_EXTRACTION_PROMPT
        assert "{data_format}" in DATA_EXTRACTION_PROMPT_ZH

    @pytest.mark.parametrize("language", ["zh-Hans", "zh-CN", "cn", "Chinese"])
    def test_zh_language_selects_zh(self, language):
        assert get_default_system_prompt(language) == DEFAULT_SYSTEM_PROMPT_ZH
        assert get_default_title_prompt(language) == DEFAULT_TITLE_PROMPT_ZH
        assert get_data_extraction_prompt(language) == DATA_EXTRACTION_PROMPT_ZH
        assert get_default_summary_prompt(language) == DEFAULT_SUMMARY_PROMPT_ZH

    @pytest.mark.parametrize("language", ["en", "de", None, ""])
    def test_other_language_selects_en(self, language):
        assert get_default_system_prompt(language) == DEFAULT_SYSTEM_PROMPT
        assert get_default_title_prompt(language) == DEFAULT_TITLE_PROMPT
        assert get_data_extraction_prompt(language) == DATA_EXTRACTION_PROMPT
        assert get_default_summary_prompt(language) == DEFAULT_SUMMARY_PROMPT

    def test_memory_reference_prompt(self):
        assert "reference" in get_memory_reference_prompt("en")
        assert "参考" in get_memory_reference_prompt("zh-Hans")


def _provider_self(language, settings_data=None):
    """A minimal stand-in for a Provider instance (no __init__ needed)."""
    domain_data = {}
    if settings_data is not None:
        domain_data["settings_entry"] = settings_data
    hass = SimpleNamespace(
        data={"llmvision": domain_data},
        config=SimpleNamespace(language=language),
    )
    return SimpleNamespace(hass=hass)


class TestProviderPromptFollowsLanguage:
    def test_no_settings_entry_follows_language(self):
        assert (
            Provider._get_system_prompt(_provider_self("zh-Hans"))
            == DEFAULT_SYSTEM_PROMPT_ZH
        )
        assert (
            Provider._get_system_prompt(_provider_self("en"))
            == DEFAULT_SYSTEM_PROMPT
        )
        assert (
            Provider._get_title_prompt(_provider_self("zh-Hans"))
            == DEFAULT_TITLE_PROMPT_ZH
        )

    def test_untouched_english_default_follows_language(self):
        # A stored value that is still the English default counts as
        # "not customized": an existing Chinese install must not stay
        # pinned to English output after upgrading to the fork.
        settings = {
            "provider": "Settings",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "title_prompt": DEFAULT_TITLE_PROMPT,
        }
        assert (
            Provider._get_system_prompt(_provider_self("zh-Hans", settings))
            == DEFAULT_SYSTEM_PROMPT_ZH
        )
        assert (
            Provider._get_title_prompt(_provider_self("zh-Hans", settings))
            == DEFAULT_TITLE_PROMPT_ZH
        )

    def test_customized_prompt_always_wins(self):
        settings = {
            "provider": "Settings",
            "system_prompt": "请用粤语回答。",
            "title_prompt": "我的标题提示词",
        }
        assert (
            Provider._get_system_prompt(_provider_self("zh-Hans", settings))
            == "请用粤语回答。"
        )
        assert (
            Provider._get_title_prompt(_provider_self("en", settings))
            == "我的标题提示词"
        )


def _flat(d, prefix=""):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flat(value, prefix + key + "."))
        else:
            out[prefix + key] = value
    return out


class TestTranslationCoverage:
    def test_cn_covers_en_config_and_services(self):
        en = json.loads((ROOT / "translations" / "en.json").read_text(encoding="utf-8"))
        cn = json.loads((ROOT / "translations" / "cn.json").read_text(encoding="utf-8"))
        zh_hans = json.loads((ROOT / "translations" / "zh-Hans.json").read_text(encoding="utf-8"))
        strings = json.loads((ROOT / "strings.json").read_text(encoding="utf-8"))
        for name, data in (("cn", cn), ("zh-Hans", zh_hans)):
            assert set(_flat(data)) == set(_flat(en)), f"{name} keys differ from en"
            assert set(_flat(data["config"])) == set(_flat(strings["config"]))
            assert set(_flat(data["services"])) == set(_flat(strings["services"]))

    def test_mistral_translated(self):
        cn = json.loads((ROOT / "translations" / "cn.json").read_text(encoding="utf-8"))
        mistral = cn["config"]["step"]["mistral"]
        assert mistral["title"] == "配置 Mistral"
        assert "Mistral" in mistral["sections"]["model_section"]["data_description"]["default_model"]

    def test_services_translated(self):
        cn = json.loads((ROOT / "translations" / "cn.json").read_text(encoding="utf-8"))
        assert cn["services"]["image_analyzer"]["name"] == "图片分析器"
        assert cn["services"]["stream_analyzer"]["fields"]["image_entity"]["name"] == "摄像头实体"
        assert cn["services"]["image_analyzer"]["fields"]["image_entity"]["name"] == "图片实体"


class TestTimelineStringsCn:
    def test_cn_categories_match_en(self):
        en = json.loads((ROOT / "timeline_strings" / "en.json").read_text(encoding="utf-8"))
        cn = json.loads((ROOT / "timeline_strings" / "cn.json").read_text(encoding="utf-8"))
        assert set(cn["categories"]) == set(en["categories"])
        for category, data in cn["categories"].items():
            assert set(data["labels"].values()) <= set(
                en["categories"][category]["labels"].values()
            )

    def test_chinese_labels_match_as_substrings(self):
        # CJK has no word boundaries: the cn.json regex template is a bare
        # key, so "人" must match inside a running Chinese sentence.
        cn = json.loads((ROOT / "timeline_strings" / "cn.json").read_text(encoding="utf-8"))
        template = cn["regex"]["pattern"]
        assert "${key}" in template
        pattern = template.replace("${key}", re.escape("人"))
        assert re.search(pattern, "画面里有一个人走过", re.IGNORECASE)
        pattern = template.replace("${key}", re.escape("快递"))
        assert re.search(pattern, "快递员放下一个包裹", re.IGNORECASE)
        assert not re.search(pattern, "画面里有一个人走过", re.IGNORECASE)
