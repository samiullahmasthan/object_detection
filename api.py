from flask import Flask,request
from flask_restx import Api, Resource
import json

app = Flask(__name__)
api = Api(app)

# Load the results from the JSON file
with open('output.json', 'r') as json_file:
    results = json.load(json_file)
    
    
    print(results, 'results')


@api.route('/query_objects')
class QueryObjects(Resource):
    @api.doc(params={'image_name': 'Name of the frame to query'})
    def get(self):
        frame_name = request.args.get('image_name')
        for result in results:
            print(result,'check')
            if result['image_name'] == frame_name:
                return {"frame_number": result['frame_number'],
                        "no_of_objects": str(result["no_of_objects"]),
                        "image_name": result["image_name"]}

        return {"error": "Frame not found"}, 404


if __name__ == '__main__':
    app.run(debug=True)
