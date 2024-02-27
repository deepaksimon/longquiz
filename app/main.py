from fastapi import FastAPI
from titanic_model.predict import make_prediction 
import json 
import numpy as np

app = FastAPI()


@app.get("/")
async def root():
    
    data_in={'PassengerId':[79],'Pclass':[2],'Name':["Caldwell, Master. Alden Gates"],'Sex':['male'],'Age':[0.83],
                'SibSp':[0],'Parch':[2],'Ticket':['248738'],'Cabin':[np.nan,],'Embarked':['S'],'Fare':[29]}
    results = make_prediction(input_data=data_in)
#    print(results['predictions'][0])
#  return json.dumps(results['predictions'])
    final_result = {"result":str(results['predictions'][0])} 
    return final_result
    