# Data dictionary

This document details the data that you will receive for this project. If you have additional questions please just ask us!

## Core fields

These are fields that you'll either need for your analysis and modelling, or that you'll want to refer to with some regularity.

| Field name               | Description |
| ------------------------ | ----------- |
| Genesys_interaction_id   | A unique identifier for the conversation in GUID format. It does not mean a unique _person_; one person returning to the site multiple times could have multiple conversations. |
| First_prompt             | The first message that the user sent to Roo during their conversation. |
| First_response           | The first message that Roo sent _after_ the First_prompt. We tried to filter out cases where it was simply a "Welcome to Roo" message. |
| First_label              | One of: `TP`, `FP`, `TN`, `FN`. Assigned by one of our team (`Labeller`). |
| Ease_of_use              | Optional user survey result; an integer 1-4, with 4 being best. |
| Helpfulness              | Optional user survey result; an integer 1-4, with 4 being best. |
| Understanding            | Optional user survey result; an integer 1-4, with 4 being best. |
| Recommendation           | Optional user survey result; an integer 1-4, with 4 being best. |

## Secondary/extra fields

These are fields that are likely not needed for your core model, but that may be of use during analysis.

| Field name               | Description |
| ------------------------ | ----------- |
| Full_conversation        | The entire conversation in JSON format. Messages from user `workflow` are Roo bot messages; messages from user `customer` are the user's inputs. |
| Flag_label_for_review    | Boolean; indicates whether the first labeller wanted a second pair of eyes on the item with respect to `First_label`. Generally speaking, `True` means that the appropriate label for the conversation is ambiguous. |
| Comment                  | Free-form text explaining the choice of `First_label`. Worth looking at if a label seems strange to you. |
| Reviewer_suggested_label | Should only be present if `Flag_label_for_review` is `True`. One of `TP`, `FP`, `TN`, `FN`. |
| Reviewer_comment         | Free-form text explaining the choice of `Reviewer_suggested_label`, though this will often be blank if the reviewer agreed with the first labeller. |

## Additional fields

These are fields that are either likely to be irrelevant except for reference with the PPFA team, or that are not likely to be useful for training or analysis.

| Field name                     | Description |
| ------------------------------ | ----------- |
| Interaction_contains_PII       | Boolean; indicates that someone included personally identifiable information in the prompt. These should all be `False` unless we missed one. We should not train with actual PII, but you're welcome to try putting in fake PII to see how Roo performs on it. |
| Provided_prompt                | Boolean (`NULL` means `False`); means that the prompt is from the Roo FAQs or general topic taxonomy, rather than something someone typed. |
| Provided_prompt_autocalculated | Boolean (`NULL` means `False`); means the same as `Provided_prompt` but additionally that this was chosen as an auto-suggestion by the chatbot. We can't always identify this one way or the other. |
| Labeller                       | The name of the person who chose `First_label`. |
| Reviewer                       | The name of the person who chose `Reviewer_suggested_label`. |
