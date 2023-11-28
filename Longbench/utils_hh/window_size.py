import torch

has_entered_function_get_sliding_windows = False 
def get_sliding_windows(mode: str, config, prompt_len: int, max_tokens: int, spec_size: int, min_size: int):
    num_hidden_layers = config.num_hidden_layers
    sliding_windows = [0 for _ in range(num_hidden_layers)]

    if mode == 'default':
        for i in range(num_hidden_layers):
            sliding_windows[i] = 4096
    elif mode == 'sp1':
        for i in range(num_hidden_layers):
            sliding_windows[i] = prompt_len + max_tokens - int(max_tokens / num_hidden_layers) * i
    elif mode == 'sp2':
        for i in range(num_hidden_layers):
            sliding_windows[i] = prompt_len + max_tokens - int(max_tokens / num_hidden_layers) * 2 * i
    elif mode == 'sp1_':
        for i in range(num_hidden_layers):
            sliding_windows[i] = prompt_len + max_tokens - int(max_tokens / num_hidden_layers) * i
        sliding_windows = sliding_windows[::-1]
    elif mode == 'sp2_':
        for i in range(num_hidden_layers):
            sliding_windows[i] = prompt_len + max_tokens - int(max_tokens / num_hidden_layers) * 2  * i
        sliding_windows = sliding_windows[::-1]
    elif mode == 'total':
        for i in range(num_hidden_layers):
            sliding_windows[i] = prompt_len + max_tokens
    elif mode == 'spec':
        for i in range(num_hidden_layers):
            sliding_windows[i] = spec_size
        if min_size != 0:
            dif = max(0, min_size - spec_size)
            dif = min(dif, spec_size - 50)
            for i in range(num_hidden_layers // 2):
                sliding_windows[i] += dif
            for i in range(num_hidden_layers // 2, num_hidden_layers):
                sliding_windows[i] -= dif
    elif mode == 'test':
        for i in range(num_hidden_layers // 2 + 5):
            sliding_windows[i] = 5000
        for i in range(num_hidden_layers // 2 + 5, num_hidden_layers):
            sliding_windows[i] = 3000

    for i in range(num_hidden_layers):
        sliding_windows[i] = max(sliding_windows[i], 10)

    global has_entered_function_get_sliding_windows
    if not has_entered_function_get_sliding_windows:
        print('window size is as follow:(mode:', mode, ')')
        print(dict(zip(range(num_hidden_layers), sliding_windows)))
        has_entered_function_get_sliding_windows = True
    return sliding_windows


def cosine_similarity(x1, x2):
    assert len(x1.shape) == 1 and x1.shape[0] == x2.shape[0]
    norm_vec1 = torch.norm(x1)
    norm_vec2 = torch.norm(x2)
    dot = torch.dot(x1, x2)
    norm = norm_vec1 * norm_vec2
    #quite naive, should be improve later
    if torch.isinf(dot) and torch.isinf(norm):
        return  torch.tensor(1.0)

    cosine_sim = dot / norm
    return cosine_sim

