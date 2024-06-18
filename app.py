import base64
import os
import pickle

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import litserve as ls
from fastapi.responses import JSONResponse

from main import get_embeddings, get_model_from_ckpt, MODEL_CKPT

AUTHORIZED_USERS = {
    "eugene",
    "szymon",
    "yc-collab",
}


class ClayModelAPI(ls.LitAPI):
    def setup(self, device):
        self.model = get_model_from_ckpt(MODEL_CKPT)

    def decode_request(self, request):
        print(request)  # todo, persist?
        return request

    def predict(self, request):
        if request["whoami"] not in AUTHORIZED_USERS:
            return JSONResponse(
                status_code=401,
                content={"reason": "Please reach out to host to authorize access"},
            )

        lat, lon = request["lat"], request["lon"]
        start, end = request["startdate"], request["enddate"]
        size = request.get("size", 256)
        embeddings, stack = get_embeddings(
            lat, lon, start, end, size=size, model=self.model
        )
        pickled_array = pickle.dumps(embeddings)
        base64_encoded_array = base64.b64encode(pickled_array).decode("utf-8")
        return {"output": base64_encoded_array}


if __name__ == "__main__":
    api = ClayModelAPI()
    server = ls.LitServer(api)
    server.run(port=8094)
