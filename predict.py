import pandas as pd
import os
from config import config as args
from create_model import function1
import pandas

file_name = 'test.pkl'
test = pd.read_pickle(os.path.join(args.finalPkl_ph, file_name))
print(test.head)