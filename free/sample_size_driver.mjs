#!/usr/bin/env node
// Node driver for the Zetyra frontend sample-size math (cluster + longitudinal).
// Consumed by free/test_cluster_rct.py and free/test_longitudinal.py.
//
// Usage: node sample_size_driver.mjs <path-to-frontend>
// Reads newline-delimited JSON requests from stdin, writes responses.

import path from "node:path";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

const frontendPath = process.argv[2];
if (!frontendPath) {
  console.error("usage: node sample_size_driver.mjs <path-to-Zetyra/frontend>");
  process.exit(2);
}

const modulePath = path.resolve(frontendPath, "src/lib/stats/sample_size.ts");
const mod = await import(pathToFileURL(modulePath).href);

const dispatch = {
  z_alpha: (r) => mod.zAlpha(r.alpha, r.twoSided),
  z_beta: (r) => mod.zBeta(r.power),
  design_effect: (r) => mod.designEffect(r.clusterSize, r.icc),
  crt_individual_n: (r) => mod.crtIndividualN(r.input),
  crt_sample_size: (r) => mod.crtSampleSize(r.input),
  slope_var_ar1: (r) => mod.slopeVarAR1(r.sd, r.rho, r.m),
  slope_var_cs: (r) => mod.slopeVarCS(r.sd, r.rho, r.m),
  longitudinal_effective_variance: (r) => mod.longitudinalEffectiveVariance(r.input),
  longitudinal_sample_size: (r) => mod.longitudinalSampleSize(r.input),
};

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  if (!line.trim()) return;
  try {
    const req = JSON.parse(line);
    const fn = dispatch[req.op];
    if (!fn) {
      process.stdout.write(JSON.stringify({ error: `unknown op: ${req.op}` }) + "\n");
      return;
    }
    const result = fn(req);
    process.stdout.write(JSON.stringify({ result }, (_k, v) => {
      if (v === Infinity) return "Infinity";
      if (v === -Infinity) return "-Infinity";
      if (Number.isNaN(v)) return "NaN";
      return v;
    }) + "\n");
  } catch (err) {
    process.stdout.write(JSON.stringify({ error: String(err) }) + "\n");
  }
});
