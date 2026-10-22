import { expect, type Page, test } from "@playwright/test";

const SHOTS = process.env.SHOT_DIR;

async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: false });
}

test("untargeted discovery, candidate viewer and hand-off to localization", async ({ page }) => {
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto("/?token=e2e");
  const datasets = (await (await page.request.get("/api/datasets")).json()) as { name: string; path: string }[];
  const run = datasets.find((d) => d.name.startsWith("PASEF_isomers"))!;
  const psmPath = run.path.replace(/PASEF_isomers\.d$/, "PASEF_isomers.psms.csv");

  await page.getByRole("button", { name: "Isomer discovery" }).click();
  await expect(page.getByRole("heading", { name: "Isomer discovery" })).toBeVisible();
  await expect(page.getByText("No discovery runs yet")).toBeVisible();
  await page.getByRole("combobox", { name: /^Dataset/ }).selectOption({ label: run.name });

  // The PSM export: pasted path, then a preview of what the importer found.
  await page.getByRole("button", { name: "Choose PSM file" }).click();
  await page.getByLabel("Or paste a path").fill(psmPath);
  await page.getByRole("button", { name: "Choose", exact: true }).click();
  await expect(page.getByText("12 PSMs")).toBeVisible();
  await expect(page.locator(".badge", { hasText: "peaks" })).toBeVisible();

  // A measured precision replaces the placeholder.
  await page.getByRole("button", { name: "Thresholds" }).click();
  await expect(page.getByText("placeholder 0.005")).toBeVisible();
  await page.getByLabel("Mobility precision").fill("0.003");
  await page.getByLabel("Mobility precision").press("Enter");
  await expect(page.getByText("placeholder 0.005")).toHaveCount(0);

  await page.getByRole("button", { name: "Run discovery" }).click();
  const table = page.locator("section", { hasText: "Candidates" }).locator("table.data");
  await expect(table.locator("tbody tr").first()).toBeVisible({ timeout: 200_000 });
  await expect(page.locator(".tile", { hasText: "High-confidence putative" }).locator(".tile-value")).not.toHaveText("0");
  await expect(table.getByText("IM-separated").first()).toBeVisible();
  await shot(page, "d1-discovery-run");

  // Best candidate first: open it.
  const first = table.locator("tbody tr").first();
  await expect(first.getByText("High-confidence putative")).toBeVisible();
  await first.click();
  const viewer = page.locator("section", { has: page.getByRole("heading", { name: /^Candidate \d+$/ }) });
  await expect(viewer.locator(".plot-eic .scatterlayer .trace").first()).toBeAttached();
  await expect(viewer.locator(".heatmaplayer .hm")).toHaveCount(1);
  await expect(viewer.getByRole("heading", { name: "MS/MS mirror plot" })).toBeVisible();
  await expect(viewer.getByText("RT difference (min)")).toBeVisible();
  await expect(viewer.getByText("LC-MS signal, not molar abundance")).toBeVisible();
  if (SHOTS) {
    await page.setViewportSize({ width: 1440, height: 2800 });
    await page.waitForTimeout(600);
    await viewer.screenshot({ path: `${SHOTS}/d2-candidate-viewer.png` });
    await page.setViewportSize({ width: 1440, height: 900 });
  }

  // Hand-off: two RT windows, the precursor and the peptide land in the localization workflow.
  await viewer.getByRole("button", { name: "Localize with CID-TIMS" }).click();
  await expect(page.getByRole("heading", { name: "Targets", exact: true })).toBeVisible();
  await expect(page.getByPlaceholder("SPQLEDEAKELQ(-0.98)")).not.toHaveValue("");
  await expect(page.locator('input[value^="precursor "]')).toHaveCount(1);
  const windows = await page.evaluate(() => JSON.parse(localStorage.getItem("lctimspy-session") ?? "{}").state?.windows?.length);
  expect(windows).toBe(2);

  expect(errors).toEqual([]);
});
