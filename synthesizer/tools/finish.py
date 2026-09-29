from .base import Tool

docs = """`Finish(response="response")`
Finish will return a natural language response to the user, along with all results from previous actions.
While the action is wrapped in backticks, do not use additional `backticks` in your "response" argument, as this will cause issues for parsing the action.
You should include any useful context, such as output filenames and whether or not you were successful, back to the user.
Please do not finish before either (1) getting output files or (2) determining you cannot complete the requested task.
"""

class Finish(Tool):
    def documentation(self):
        return docs
    
    def __call__(self, **params):
        if "response" not in params:
            return 'Please specify a double-quoted response using the explicit keyword response="Natural language response..."', '⚠️Failed, no response="Natural language response..."'
        self.setup_config["agent"].finished = True
        self.setup_config["agent"].response = params["response"]
        return None, "🏁🏁🏁"


        
