import cherrypy
import json


class Root:

    def __init__(self):
        self.best_fit = -1
        self.best_ind = {}


    @cherrypy.expose
    def index(self):
        if not self.best_fit == -1:
            return "Current Best: " + \
                "Fitness: " + str(self.best_ind["fitness"]) +  "\n" + \
                "Params: " + str(self.best_ind["params"])
        return "No data is available. Please run the see-segment container."

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def update(self):
        result = {"operation": "request", "result": "success"}

        input_json = cherrypy.request.json
        input_json = json.loads(input_json)


        if self.best_fit == -1 or self.best_ind["fitness"] > input_json["fitness"]:
            self.best_ind = input_json
            self.best_fit = self.best_ind["fitness"]
        return result


cherrypy.quickstart(Root(), '/', {'global': {'server.socket_host':'0.0.0.0','server.socket_port': 8181}})