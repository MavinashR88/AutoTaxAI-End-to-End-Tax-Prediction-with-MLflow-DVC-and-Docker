from src.data_pre.DataProcessing import Preprocessing

pipeline = Preprocessing("data/raw/HousingData.csv")
result = pipeline.run_pipeline()

