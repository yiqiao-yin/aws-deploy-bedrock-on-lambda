# 🚀 AWS Bedrock Lambda API Deployment Guide

This guide walks you through setting up an **AWS Lambda function** that invokes **Amazon Titan** on **AWS Bedrock**, deploys it with **API Gateway**, and allows invoking it from external scripts in **Python, Node.js, and Swift**.

---

## 📌 **Step 1: Prepare Lambda Function**
Create a new **AWS Lambda function** and paste the following **Python script** into the Lambda editor.

### **Lambda Function (Python)**
```python
import boto3
import json

# Initialize AWS Bedrock Runtime client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")  # Ensure correct region

def lambda_handler(event, context):
    """
    AWS Lambda function to invoke Amazon Titan via AWS Bedrock.
    Allows users to specify maxTokenCount, temperature, and topP in the request payload.
    """

    # Extract request body (handling API Gateway string-wrapped JSON)
    try:
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])  
        else:
            body = event.get("body", {})

        # Extract parameters with default values
        prompt = body.get("prompt", "What is the meaning of life?")
        max_token_count = body.get("maxTokenCount", 200)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("topP", 0.9)

    except (json.JSONDecodeError, TypeError):
        prompt, max_token_count, temperature, top_p = "What is the meaning of life?", 200, 0.7, 0.9

    # Construct request to Amazon Titan
    titan_body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_token_count,
            "temperature": temperature,
            "topP": top_p
        }
    })

    kwargs = {
        "modelId": "amazon.titan-tg1-large", # https://docs.aws.amazon.com/bedrock/latest/userguide/titan-text-models.html
        "contentType": "application/json",
        "accept": "application/json",
        "body": titan_body
    }

    try:
        resp = bedrock_runtime.invoke_model(**kwargs)
        resp_json = json.loads(resp["body"].read().decode("utf-8"))

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Amazon Titan Response",
                "input_prompt": prompt,
                "parameters_used": {
                    "maxTokenCount": max_token_count,
                    "temperature": temperature,
                    "topP": top_p
                },
                "model_response": resp_json
            }, indent=2)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
```

---

## 📌 **Step 2: Test in Lambda Console**
In the **AWS Lambda Console**, test the function with the following JSON payload:

```json
{
    "prompt": "What is 1+2?",
    "maxTokenCount": 10,
    "temperature": 0.5,
    "topP": 0.8
}
```

---

## 📌 **Step 3: Install Latest Bedrock Layer (Optional)**
If you need the latest **boto3**, follow these steps:

```sh
mkdir ./bedrock-layer
cd ./bedrock-layer
mkdir ./python
pip3 install -t ./python/ boto3
zip -r bedrock-layer.zip .
aws lambda publish-layer-version --layer-name bedrock-layer --zip-file fileb://bedrock-layer.zip
```

- This creates a **Lambda layer**.
- Copy the **ARN** of the new layer from AWS Lambda → Layers.
- Navigate to your Lambda function, go to **Layers**, and **add by ARN**.

---

## 📌 **Step 4: Deploy API Gateway**
1. **Go to AWS Lambda Console → API Gateway**.
2. **Create a new API Gateway** (REST API).
3. **Create a new Resource and Method (e.g., POST, GET, or ANY).**
4. **Deploy API** and note down the **Invoke URL**.

---

## 📌 **Step 5: Invoke API via Python**
Once the API is deployed, use this **Python script** to invoke it.

### **Python Invoke Script**
```python
import requests
import json

# API Gateway Invoke URL
API_URL = "https://f0j29zsaai.execute-api.us-east-1.amazonaws.com/dev/test_bedrock_v2"

# Define the payload
payload = {
    "prompt": "What is 1+2?",
    "maxTokenCount": 10,
    "temperature": 0.5,
    "topP": 0.8
}

# Headers
headers = {"Content-Type": "application/json"}

# Send POST request
response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    full_response = response.json()
    with open("full_response.json", "w") as f:
        json.dump(full_response, f, indent=2)
    
    output_text = full_response.get("model_response", {}).get("results", [{}])[0].get("outputText", "No output found.")
    print("\nExtracted Output Text:\n", output_text)
else:
    print(f"Error {response.status_code}: {response.text}")
```

---

## 📌 **Step 6: Invoke API via Node.js**
For Node.js, use the following script:

### **Node.js Invoke Script (`invoke.js`)**
```javascript
const axios = require("axios");

const API_URL = "https://f0j29zsaai.execute-api.us-east-1.amazonaws.com/dev/test_bedrock_v2";

const payload = {
    prompt: "What is 1+2?",
    maxTokenCount: 10,
    temperature: 0.5,
    topP: 0.8
};

axios.post(API_URL, payload, { headers: { "Content-Type": "application/json" } })
    .then(response => {
        console.log("\nExtracted Output Text:\n", response.data.model_response.results[0].outputText);
    })
    .catch(error => {
        console.error("API Request Failed:", error.response ? error.response.data : error.message);
    });
```

- Install dependencies using:
  ```sh
  npm install axios
  ```
- Run using:
  ```sh
  node invoke.js
  ```

---

## 📌 **Step 7: Invoke API via Swift**
For **Swift**, use the following code:

### **Swift Invoke Script (`InvokeAPI.swift`)**
```swift
import Foundation

let apiUrl = URL(string: "https://f0j29zsaai.execute-api.us-east-1.amazonaws.com/dev/test_bedrock_v2")!

var request = URLRequest(url: apiUrl)
request.httpMethod = "POST"
request.addValue("application/json", forHTTPHeaderField: "Content-Type")

let payload: [String: Any] = [
    "prompt": "What is 1+2?",
    "maxTokenCount": 10,
    "temperature": 0.5,
    "topP": 0.8
]

request.httpBody = try? JSONSerialization.data(withJSONObject: payload)

let task = URLSession.shared.dataTask(with: request) { data, response, error in
    guard let data = data, error == nil else {
        print("Request Failed:", error?.localizedDescription ?? "Unknown error")
        return
    }

    if let jsonResponse = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
       let results = jsonResponse["model_response"] as? [String: Any],
       let outputText = (results["results"] as? [[String: Any]])?.first?["outputText"] as? String {
        print("\nExtracted Output Text:\n\(outputText)")
    } else {
        print("Invalid API Response")
    }
}

task.resume()
```

- Run in **Xcode Playground** or **Swift REPL**.

---

## 🎯 **Final Thoughts**
Now, you have:
- A **working AWS Lambda function**.
- **API Gateway for invoking the function**.
- **Invoke scripts in Python, Node.js, and Swift**.

🚀 Happy Coding! 🎉
