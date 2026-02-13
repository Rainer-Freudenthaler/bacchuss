Alternative modes
================

# A few notes on alternative modes

In this vignette I am presenting a few alternative modes added to
bacchuss. They apply to epithet-functions (e.g. liknites, briseus,
isodates) as well unless noted differently.

## LiteLLM mode

University of Mannheim is using LiteLLM as the API. It uses
OpenAI-formatting for API calls instead of ollama. If your host uses
that format, add host url and api_key and choose backend = “litellm”.

``` r
library(bacchuss)

litellm_mode <- bacchuss(example_set, input_column = "paragraphs",
                        instructions = example_instructions,
                        reminder = example_reminder,
                        model = "mistral-nemo",
                        host = NULL,
                        api_key = NULL,
                        backend = "litellm")

View(litellm_mode)
```

## JSON mode

University of Mannheim is using LiteLLM as the API. It uses
OpenAI-formatting for API calls instead of ollama. If your host uses
that format, add host url and api_key and choose backend = “litellm”.

``` r
library(bacchuss)

json_mode <- bacchuss(example_set, input_column = "paragraphs",
                            instructions = example_instructions_expl_json,
                            examples = example_fewshot$input,
                            explanations = example_fewshot$explanation,
                            codes = example_fewshot$label,
                            reminder = example_reminder_expl,
                            model = "mistral-nemo",
                            host = NULL,
                            api_key = NULL,
                      expected_response_format = "json")

View(json_mode)
```
