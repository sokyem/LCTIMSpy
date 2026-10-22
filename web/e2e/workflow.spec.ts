import { expect, type Page, test } from "@playwright/test";

const SHOTS = process.env.SHOT_DIR;

async function shot(page: Page, name: string) {
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/${name}.png`, fullPage: false });
}

test("full isomer workflow in the browser", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));

  // Without the session cookie the API refuses; opening the printed URL
  // authenticates and strips the token from the address bar.
  expect((await page.request.get("/api/datasets")).status()).toBe(401);
  await page.goto("/?token=e2e");
  await expect(page).toHaveURL(/127\.0\.0\.1:8799\/$/);

  // ---- 1. datasets
  await expect(page.getByRole("heading", { name: "Datasets" })).toBeVisible();
  await expect(page.getByText("Extracted", { exact: true })).toHaveCount(4); // 3 replicates + the PASEF discovery run
  await expect(page.getByText("Not extracted")).toHaveCount(1);
  for (const name of ["RH5956", "RH5957", "RH5958"]) await page.getByLabel(`Analyse ${name}`).check();
  await expect(page.getByText("3 selected · 3 ready to analyse")).toBeVisible();
  await shot(page, "1-datasets-light");
  await page.getByRole("button", { name: "Next: choose targets" }).click();

  // ---- 2. targets
  await expect(page.getByRole("heading", { name: "Targets", exact: true })).toBeVisible();
  await page.getByPlaceholder("500.25, 700.40").fill("500.25, 700.40");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.getByText("2 of 8 plotted")).toBeVisible();
  await page.getByPlaceholder("SPQLEDEAKELQ(-0.98)").fill("SPQLEDEAKELQ");
  await page.getByRole("button", { name: "Calculate" }).click();
  await expect(page.getByRole("cell", { name: "b1 (1+)", exact: true })).toBeVisible();
  await shot(page, "2-targets-light");
  await page.getByRole("button", { name: "Next: set RT windows" }).click();

  // ---- 3. RT windows: two peaks are detected automatically
  const eic = page.locator(".plot-eic");
  await expect(eic.locator(".scatterlayer .trace")).toHaveCount(2);
  const w1start = page.getByLabel("W1 start (min)");
  const w2start = page.getByLabel("W2 start (min)");
  await expect(w1start).toBeVisible();
  await expect(w2start).toBeVisible();
  expect(Number(await w1start.inputValue())).toBeGreaterThan(0.3);
  expect(Number(await w1start.inputValue())).toBeLessThan(0.4);
  expect(Number(await w2start.inputValue())).toBeGreaterThan(0.7);
  const mob = page.locator(".plot-mob");
  await expect(mob.locator(".scatterlayer .trace")).toHaveCount(4); // 2 targets x 2 windows
  await shot(page, "3-windows-light");

  // Synthetic Plotly edit (what dragging a band edge emits): W1 end -> 0.47.
  await eic.evaluate((el) =>
    (el as unknown as { emit: (n: string, p: object) => void }).emit("plotly_relayout", { "shapes[0].x1": 0.47 }),
  );
  await expect(page.getByLabel("W1 end (min)")).toHaveValue("0.47");

  // Typing a bound updates the chart's band.
  await page.getByLabel("W2 end (min)").fill("0.88");
  await page.getByLabel("W2 end (min)").press("Enter");
  await expect(page.getByLabel("W2 end (min)")).toHaveValue("0.88");

  // A real mouse drag redraws W1 through the box-select gesture.
  await page.getByRole("button", { name: "Redraw" }).first().click();
  const plotArea = eic.locator(".nsewdrag").first();
  const box = (await plotArea.boundingBox())!;
  const xAt = async (rt: number) => {
    const [lo, hi] = await eic.evaluate(
      (el) => (el as unknown as { _fullLayout: { xaxis: { range: number[] } } })._fullLayout.xaxis.range,
    );
    return box.x + ((rt - lo) / (hi - lo)) * box.width;
  };
  await page.mouse.move(await xAt(0.34), box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(await xAt(0.42), box.y + box.height / 2, { steps: 8 });
  await page.mouse.move(await xAt(0.46), box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  // Starts just left of the old W1 band: an armed redraw must not be captured
  // by the band's drag handles. Tolerance 0.005 min is ~4 px at this zoom.
  await expect.poll(async () => Number(await w1start.inputValue())).toBeCloseTo(0.34, 2);
  await expect.poll(async () => Number(await page.getByLabel("W1 end (min)").inputValue())).toBeCloseTo(0.46, 2);
  await expect(page.locator('.wchip[data-armed="true"]')).toHaveCount(0);

  // ---- 4. results
  await page.getByRole("button", { name: "Next: results" }).click();
  await expect(page.getByRole("heading", { name: "Results" })).toBeVisible();
  await expect(page.locator(".tile").filter({ hasText: "Significant pairs" })).toContainText("1 / 1");
  await expect(page.locator("section", { hasText: "Per-sample mobility" }).locator("tbody tr")).toHaveCount(6);
  await expect(page.locator(".plot-box .boxlayer .trace")).toHaveCount(2);
  await shot(page, "4-results-light");

  // Export: check the response the button fetches. (Waiting on Edge's download
  // event leaves state behind that errors in Playwright's worker teardown.)
  const exportResponse = page.waitForResponse((r) => r.url().endsWith("/api/export"));
  await page.getByRole("button", { name: "Differences CSV" }).click();
  const r = await exportResponse;
  expect(r.status()).toBe(200);
  // (The CSV body itself is checked in tests/api; the page consumes it here.)
  expect(r.headers()["content-disposition"]).toContain("lctimspy_mobility-difference.csv");

  // ---- dark theme
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("group", { name: "Theme" }).getByRole("button", { name: "Dark" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await shot(page, "5-settings-dark");
  await page.getByRole("button", { name: /RT windows/ }).click();
  await expect(mob.locator(".scatterlayer .trace")).toHaveCount(4);
  await shot(page, "6-windows-dark");
  await page.getByRole("button", { name: /Results/ }).click();
  await expect(page.locator(".plot-box .boxlayer .trace")).toHaveCount(2);
  await shot(page, "7-results-dark");
  await page.getByRole("button", { name: /Datasets/ }).click();
  await shot(page, "8-datasets-dark");

  expect(errors).toEqual([]);
});
