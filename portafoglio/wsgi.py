from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app import server as main_server
from frontiera_app import frontier_server

application = DispatcherMiddleware(main_server, {
    '/frontiera-efficiente': frontier_server,
})
