import mlflow

model_uri = "models:/raymond_zhou.mlflow3.openai/1"
openai_agent = mlflow.pyfunc.load_model(model_uri)

def openai_app(question: str) -> dict:
  response = openai_agent.predict(question)
  response_text = response[0].strip()
  return {"content": response_text}
