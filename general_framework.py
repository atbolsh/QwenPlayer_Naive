# This file should have all the code shared between all (or most) of the tasks
# should have all the torch libraries I need
from visual_transformer import *
from visual_transformer.enhanced_model import *
import random

#device = torch.device('cuda:0') # CHANGE THIS EVERY TIME
device = torch.device('cuda:1') # CHANGE THIS EVERY TIME

from game import *

game_settings = BIG_tool_use_advanced_2_5
game_settings.gameSize = 224 # for compatibility with brain's expected size
G = discreteGame(game_settings)

########

vocab_size = 10000
# tokenizer.save_model(".", "tokenizer/eng_sentences_tokenizer_vc10000")
tokenizer = ByteLevelBPETokenizer(
    "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-vocab.json",
    "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-merges.txt",
)   
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=32)
tokenizer.enable_padding()

# to override default behavior: make sure you always set 'skip_special_tokens' to false when using decoder
tokenizer.add_special_tokens(['<forward>', '<clock>', '<anticlock>'])

## Dataset
class SampleDataset(Dataset):
    def __init__(self, seq_length = 32, evaluate: bool = False, tokenizer=None, device = None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.seq_length = seq_length
        if tokenizer is None:
            tokenizer = ByteLevelBPETokenizer(
                "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-vocab.json",
                "./text_pretraining_tokenizer/eng_sentences_tokenizer_vc10000_v2-merges.txt",
            )   
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )   
        tokenizer.enable_truncation(max_length=self.seq_length)
        tokenizer.enable_padding()#length=seq_length)
        # or use the RobertaTokenizer from `transformers` directly.
        tokenizer.add_special_tokens(['<forward>', '<clock>', '<anticlock>'])

        self.examples = []

        src_files = Path("./text_pretraining_data/").glob("*-eval.txt") if evaluate else Path("./text_pretraining_data/").glob("*-train.txt")
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i): 
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i]).to(self.device)

sdt = SampleDataset(tokenizer=tokenizer)
sdv = SampleDataset(tokenizer=tokenizer, evaluate=True)
num_controls = len(sdt)

ent_criterion = nn.CrossEntropyLoss(ignore_index=0)

def get_text_loss(res, inputs):
    return torch.sum(ent_criterion(res[:, :, :-1], inputs[:, 1:]))

########

img_criterion = nn.MSELoss()

########

def get_settings_batch(batch_size):
    return [G.random_bare_settings(gameSize=224, max_agent_offset=0.5) for i in range(batch_size)]

########

def get_images(settings_batch, device=device, ignore_agent=False, ignore_gold=False, ignore_walls=False):
    batch_size = len(settings_batch)
    img = torch.zeros(batch_size, 224, 224, 3)
    should_draw = (ignore_agent or ignore_gold or ignore_walls)
    for i in range(batch_size):
        G2 = discreteGame(settings_batch[i])
        if should_draw:
            G2.draw(ignore_agent=ignore_agent, ignore_gold=ignore_gold, ignore_wals=ignore_walls)
        img[i] = torch.tensor(G2.getData())
    img = torch.permute(img, (0, 3, 1, 2)).contiguous().to(device)
    return img


