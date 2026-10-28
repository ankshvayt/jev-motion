"""Token, price and time estimates. All of them are estimates made before sending.

Jev counts about pixels / 350 + 30 tokens per image and about chars / 4 for text; it costs $0.042 per million input
tokens. Decider is serial and free; the estimate counts about 64 ms per text question and, with an image, 0.82 s for the
first question and about 446 ms for each extra one.
"""
QUESTION_BUDGET = 26_000
QUESTION_CAP = 32_000
REQUEST_CAP = 64_000
JEV_USD_PER_M_INPUT = 0.042

LOCAL_TEXT_S = 0.064
LOCAL_IMAGE_FIRST_S = 0.82
LOCAL_IMAGE_EXTRA_S = 0.446


def text_tokens(chars):
    return -(-chars // 4)


def image_tokens(w, h):
    return -(-(w * h) // 350) + 30


def request_tokens(text_chars, image_dims):
    """Estimated input tokens of one request: its text plus every `(w, h)` image as sent."""
    return text_tokens(text_chars) + sum(image_tokens(w, h) for w, h in image_dims)


def usd(tokens):
    return tokens * JEV_USD_PER_M_INPUT / 1e6


def fmt_usd(amount):
    """A dollar amount for a message: `$0.000` when exactly zero, `<$0.001` when above zero but under a tenth of a
    cent, else three decimals (`$0.042`)."""
    if amount > 0 and amount < 0.001:
        return "<$0.001"
    return f"${amount:.3f}"


def local_seconds(n_questions, n_images):
    """Estimated serial Decider time for one request with `n_questions` questions and `n_images` images."""
    if n_questions <= 0:
        return 0.0
    if n_images <= 0:
        return n_questions * LOCAL_TEXT_S
    return LOCAL_IMAGE_FIRST_S + (n_questions - 1) * LOCAL_IMAGE_EXTRA_S


def chunk(items, cost, budget=QUESTION_BUDGET, overhead=0):
    """Split `items` in order into groups whose summed `cost(item)` plus `overhead` stays within `budget`.
    An item that does not fit even alone gets a group of its own; nothing is dropped."""
    groups, current, used = [], [], overhead
    for item in items:
        c = cost(item)
        if current and used + c > budget:
            groups.append(current)
            current, used = [], overhead
        current.append(item)
        used += c
    if current:
        groups.append(current)
    return groups
