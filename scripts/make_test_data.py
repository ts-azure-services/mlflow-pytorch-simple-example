import json
import codecs
from torch import tensor

test_features = tensor([[5.0], [8.1]]).numpy()
test_features = test_features.tolist()

input_data = {"input_data": test_features}

input_data = json.dump(
        input_data,
        codecs.open("./data/request.json", 'w', encoding='utf-8')
        )
