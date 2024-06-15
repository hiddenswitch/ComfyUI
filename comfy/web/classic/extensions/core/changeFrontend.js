import { app } from "../../scripts/app.js";

const id = "Comfy.Frontend";
const ext = {
	name: id,
	async setup(app) {
		app.ui.settings.addSetting({
			id,
			name: "Client Frontend",
			defaultValue: "classic",
			type: "combo",
			options: ["classic", "sabre"],
			onChange(value) {
				if (value !== app.ui.settings.getSettingValue(id, "classic")) {
					setTimeout(() => {
						window.location.reload();
					}, 10);
				}
			},
		});
	},
};

app.registerExtension(ext);
