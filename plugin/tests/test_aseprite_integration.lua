-- Run in Aseprite's real scripting runtime, from the project root:
-- aseprite --batch --script plugin/tests/test_aseprite_integration.lua
-- First run: .venv/bin/python -m server.macos_check
local pluginDir = app.params.plugin_dir or "plugin"
local output = app.params.validation_dir or "output/macos-validation"
local sprites = assert(loadfile(app.fs.joinPath(pluginDir, "sprite.lua")))(pluginDir)
local f = assert(io.open(app.fs.joinPath(output, "aseprite-payload.json"), "r"))
local data = json.decode(f:read("a")); f:close()
assert(#data.images == 4, "run all four model tasks first")
local spr = Sprite(64, 64, ColorMode.RGB)
spr.layers[1]:cel(1).image:drawPixel(0, 0, app.pixelColor.rgba(12, 34, 56, 255))
local original = spr.layers[1]:cel(1).image.bytes
for n, spec in ipairs(data.images) do
  local img = sprites.imageFromPayload(spec, n)
  assert(img.width == 64 and img.height == 64)
  local entry = sprites.insertAsLayer(img, "SpriteLoom check " .. n, false)
  assert(entry.layer:cel(1).image.bytes == img.bytes)
end
assert(#spr.layers == 5)
assert(spr.layers[1]:cel(1).image.bytes == original, "source layer was changed")
spr:saveAs(app.fs.joinPath(output, "integration.aseprite"))
print("PASS: 4 real model results decoded and inserted; source layer unchanged; 5 layers saved")
