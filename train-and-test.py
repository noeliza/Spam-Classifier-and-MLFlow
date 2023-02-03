#%%
import warnings
from datasets import load_dataset
from utils.pipeline import preprocessor, custom_pipeline


warnings.filterwarnings('ignore')


#%%
df = load_dataset('sms_spam')['train'].to_pandas()

#%%
# initialize preprocessing
preprocess_data = preprocessor(df, 'sms', 'label')
preprocess_data.unique()
preprocess_data.label_summary()

# split the data
preprocess_data.split_data(random_state = 5)


pipeline = custom_pipeline(
    preprocess_data.X_train, 
    preprocess_data.y_train,
    preprocess_data.X_test,
    preprocess_data.y_test,
    n_components = 200, 
    random_state = 0, 
    average = 'macro', 
    tune_scoring = 'f1_macro', 
    champ_scoring = 'test F1',
    n_jobs = 8, 
    kfold_splits = 5, 
    trials = 50
)

pipeline.run()
