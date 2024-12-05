# DITeTriO

module load python/3.11
module load pytorch/2.0.1

run preprocessor:
go to DITeTriOPreProcessor directory and do ```dotnet run```
compile executable: ``````


evaluate:
source tetr_env/bin/activate
python -m scripts.evaluate_model --model_dir saved_models/1_padding --data data/processed_replays/players/sodiumoverdose/6657e2e7cdcf03ad6260a6d8_p1_r0.csv
