# Text rewriter

Using [LangChain](https://github.com/hwchase17/langchain) and GPT3 to rewrite a large text chunk by chunk (remembering previous modifications to take them into account).

## Theory

The text is sliced into chunks which are processed in order with the following prompt (shortened for the first two calls):

```
You are rewriting a novel, rewrite the full extract according to the instructions:

INSTRUCTIONS:
{instructions}

EXAMPLES OF REWRITINGS FROM OTHER PARTS OF THE TEXT:

{previous_rewrites}

EXTRACT TO REWRITE:
{previous_chunk}
{chunk}

REWRITTEN EXTRACT:
{previous_rewrite}
```

`instructions` is the user-provided prompt
`previous_rewrites` are examples of rewrites whose chunk is similar to the provided chunk
`previous_chunk`/`previous_rewrite` is the previous chunk of text (insuring continuity)
`chunk` is the current chunk of text

## Usage

* Source the `mentalism_index` conda env to get the dependencies
* Put the  OpenAI API key in path
* put your input text and prompt in the `data` folder (it contains examples of prompts and results)
* Run `rewrite.py`.

I recommend testing your prompts on smaller texts as it can take many trials before finding a good prompt.

## Potential improvements

* One could improve text size management to try and run on large text chunks while minimizing the risk of overflow (we could provide exactly as many examples as needed)
* let users add example rewritings to the search index in advance
* saving intermediate results to avoid loss and, overall, organize experiments
* add a best-of-3 interactive mode (better: we could let the user write their favorite version as an optional 4th option)