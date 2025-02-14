import boto3
import json

# Initialize AWS Bedrock Runtime client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")  # Ensure correct region

def lambda_handler(event, context):
    """
    AWS Lambda function to invoke Amazon Titan via AWS Bedrock.
    Allows users to specify maxTokenCount, temperature, and topP in the request payload.
    """

    # DEBUG: Print event to check incoming request structure
    print("Received event:", json.dumps(event, indent=2))

    # Extract request body (handling API Gateway string-wrapped JSON)
    try:
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])  
        else:
            body = event.get("body", {})

        # Extract prompt and optional parameters with defaults
        prompt = body.get("prompt", "What is the meaning of life?")
        max_token_count = body.get("maxTokenCount", 200)  # Default to 200 tokens
        temperature = body.get("temperature", 0.7)  # Default to 0.7 (moderate randomness)
        top_p = body.get("topP", 0.9)  # Default to 0.9 (controls diversity of output)

    except (json.JSONDecodeError, TypeError):
        # Handle cases where JSON is malformed
        prompt = "What is the meaning of life?"
        max_token_count = 200
        temperature = 0.7
        top_p = 0.9

    # Construct the Titan API request body with user-defined or default parameters
    titan_body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_token_count,
            "temperature": temperature,
            "topP": top_p
        }
    })

    kwargs = {
        "modelId": "amazon.titan-tg1-large",  # Titan model ID
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
