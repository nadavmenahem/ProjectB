import optuna

study = optuna.load_study(
    study_name="spectral_gcn_tuning",
    storage="sqlite:///optuna.db"
)

print("Best validation metric :", study.best_value)
print("Best hyper-parameters  :")
for k, v in study.best_trial.params.items():
    print(f"  {k:12} = {v}")

# optional: full table
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print(df.head())