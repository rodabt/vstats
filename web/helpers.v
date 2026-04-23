module main

import veb
import json

struct ApiError {
pub:
	error string
}

fn api_error(mut ctx Context, msg string) veb.Result {
	ctx.res.set_status(.bad_request)
	return ctx.json(ApiError{ error: msg })
}

fn encode_json[T](val T) string {
	return json.encode(val)
}
