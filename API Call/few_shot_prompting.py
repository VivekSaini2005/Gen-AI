from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyBA9K_HbU2JV1SLio9Mr-WKcJ3WRs6bUnc")
prompt = """You are the expert of Competetive Programming and give answer only and only related to coding dought and queries.
            
            Rule:
             - strictly follow the output in json format

             Output:
             {{
                "code":"string" or None,
                "isCodingQuestiong":boolean
             }}

            Example:
            Q. Give me a joke
            A. {{"code":null, "isCodingQuestion":false}}
            
            Q. Write a code to add two numbers in cpp.
            A. {{"code":"#include<iostream>
                using namespace std;
                int main(){
                    int a,b;
                    cin>>a>>b;
                    cout<<a+b<<endl;
                    return 0;
                }",
                "isCodingQuestion":true
                }}
            """
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction=prompt),
    contents="Write a code of prefixsum array in python"
)

print(response.text)