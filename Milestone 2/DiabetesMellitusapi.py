from lib2to3.pgen2.token import NEWLINE
from flask import Flask, request, render_template,abort
from flask_restful import Api, Resource, abort, reqparse
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
    #def get_sent_info(objects=None):
   #     objects = [0] * 10
  #      testint = 10
  #      for x in range(testint):
  #          anObject = MyObject()
   #         objects[x] = anObject
   ##         objects[x].number = x
      #  return objects
#

things = {
"name": "bruh",
"year": 1993,
"size": "200"
}

thing_post_args = reqparse.RequestParser()
thing_post_args.add_argument("name", type=str, help="Enter name", required=True)
thing_post_args.add_argument("year", type=str, help="Enter value", required=True)
thing_post_args.add_argument("size", type=str, help="Enter value", required=True)


class HelloNumbers(Resource):
    def get(self, number):
        retstring = ""
        for x in range(number):
            retstring = retstring + " " + str(x+1) + "\n"
        return retstring

class Things(Resource):
    def get(self, thing_id):
        return things
    def post(self, thing_id):
        args = thing_post_args.parse_args()
        if thing_id in things:
            abort(489, "Thing's already in list")
        things[thing_id] = {"name":args["name"], "year":args["year"], "size":args["size"]}
        return things[thing_id]

api.add_resource(HelloWorld,'/helloworld')
api.add_resource(HelloNumbers,'/helloworld/<int:number>')
api.add_resource(Things,'/todos/<int:thing_id>')
#api.add_resource(HelloNumbers,'/helloworld/<int:number>')

#@app.route("/", methods = [""])
#def index():
    #objects = get_sent_info()
# #   return render_template("index.hmtl", objects=objects)


if __name__ == '__main__':
    app.run(debug=True)