import os
import json
from aiohttp import web


class AppSettings():
    def __init__(self, user_manager):
        self.user_manager = user_manager

    def get_user_settings(self, user):
        file = self.user_manager.get_user_filepath(user, "comfy.settings.json")
        if os.path.isfile(file):
            with open(file) as f:
                return json.load(f)
        else:
            return {}

    def save_user_settings(self, user, settings):
        file = self.user_manager.get_user_filepath(user, "comfy.settings.json")
        with open(file, "w") as f:
            f.write(json.dumps(settings, indent=4))

    def add_routes(self, routes):
        @routes.get("/settings")
        async def get_settings(request):
            return web.json_response(self.get_user_settings(self.user_manager.get_request_user_id(request)))

        @routes.get("/settings/{id}")
        async def get_setting(request):
            value = None
            settings = self.get_user_settings(self.user_manager.get_request_user_id(request))
            setting_id = request.match_info.get("id", None)
            if setting_id and setting_id in settings:
                value = settings[setting_id]
            return web.json_response(value)

        @routes.post("/settings")
        async def post_settings(request):
            settings = self.get_user_settings(self.user_manager.get_request_user_id(request))
            new_settings = await request.json()
            self.save_user_settings(self.get_request_user_id(request), {**settings, **new_settings})
            return web.Response(status=200)

        @routes.post("/settings/{id}")
        async def post_setting(request):
            setting_id = request.match_info.get("id", None)
            if not setting_id:
                return web.Response(status=400)
            settings = self.get_user_settings(self.user_manager.get_request_user_id(request))
            settings[setting_id] = await request.json()
            self.save_user_settings(self.user_manager.get_request_user_id(request), settings)
            if setting_id == "Comfy.Frontend":
                self.user_manager.server.update_frontend(settings[setting_id])
            return web.Response(status=200)