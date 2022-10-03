from flask import Flask,request
from flask_restful import Api, Resource, reqparse


class MyObject:
	def __init__(self):
		self.name = "placeholder"
		self.number = 0

		