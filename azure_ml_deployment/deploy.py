from azureml.core import Workspace, Model, Environment, InferenceConfig
from azureml.core.webservice import AciWebservice

# 1️⃣ Load workspace
ws = Workspace.from_config()

# 2️⃣ Register model
model = Model.register(workspace=ws,
                       model_path="diabetes_pipeline.joblib",
                       model_name="diabetes_pipeline")

# 3️⃣ Create environment from requirements.txt
env = Environment.from_pip_requirements(
    name="diabetes-env",
    file_path="requirements.txt"
)

# 4️⃣ Create inference configuration
inference_config = InferenceConfig(entry_script="azml_score.py", environment=env)

# 5️⃣ Define deployment config
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# 6️⃣ Deploy to Azure Container Instance (ACI)
service = Model.deploy(
    workspace=ws,
    name="diabetes-predictor",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)

print("✅ Deployment complete!")
print("State:", service.state)
print("Scoring URI:", service.scoring_uri)
