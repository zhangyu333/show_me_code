# coding=utf-8
# Created : 2022/12/30 13:59
# Author  : Zy
import routes
from application import app
from gevent.pywsgi import WSGIServer

app_routes = routes

if __name__ == '__main__':
    http_server = WSGIServer(
        (
            app.config.get("SERVER_HOST"),
            app.config.get("SERVER_PORT")
        ),
        app
    )
    http_server.serve_forever()
