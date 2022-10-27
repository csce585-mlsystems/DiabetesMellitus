from flask import Flask, request, render_template
from flask_restful import Api, Resource, reqparse
import numpy as np

app = Flask(__name__)
api = Api(app)


class MyObject:
	def __init__(self):
		self.name = "placeholder"
		self.number = 0
        

# localhost:5000/helloworld
class HelloWorld(Resource):
    def get(self):
        return {'data': 'Hello, World!'}
def get_sent_info(objects=None):
    testint = 10
    for x in range(testint):
        anObject = MyObject()
        objects[x] = anObject
        objects[x].number = x
    return objects

api.add_resource(HelloWorld,'/helloworld')
#api.add_resource(HelloName,'/helloworld/<string:name>')

@app.route("/", methods = [""])
def index():
    objects = get_sent_info()
    return render_template("index.hmtl", objects=objects)


if __name__ == '__main__':
    app.run(debug=True)