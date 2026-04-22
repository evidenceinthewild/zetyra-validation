#!/usr/bin/env node
// Node driver for exercising the Zetyra frontend chi-square math module
// directly (no Python mirror). Consumed by free/test_chi_square.py.
//
// Usage: node chi_square_driver.mjs <path-to-frontend>
//   Reads newline-delimited JSON requests from stdin.
//   Emits newline-delimited JSON responses to stdout.
//
// Request schema:
//   { "op": "chi_square_p_value", "x": number, "df": number }
//   { "op": "chi_square_critical", "alpha": number, "df": number }
//   { "op": "fisher_exact_2x2", "a": int, "b": int, "c": int, "d": int }
//   { "op": "pearson_chi_square", "table": number[][] }
//   { "op": "mcnemar_classical", "a": int, "b": int, "c": int, "d": int }
//   { "op": "normal_quantile", "p": number }
//
// Response: the function's return value (number, object, or null), JSON-encoded.

import path from "node:path";
import readline from "node:readline";
import { pathToFileURL } from "node:url";

const frontendPath = process.argv[2];
if (!frontendPath) {
  console.error("usage: node chi_square_driver.mjs <path-to-Zetyra/frontend>");
  process.exit(2);
}

const modulePath = path.resolve(frontendPath, "src/lib/stats/chi_square.ts");
const mod = await import(pathToFileURL(modulePath).href);

const dispatch = {
  chi_square_p_value: (r) => mod.chiSquarePValue(r.x, r.df),
  chi_square_critical: (r) => mod.chiSquareCritical(r.alpha, r.df),
  fisher_exact_2x2: (r) => mod.fisherExact2x2(r.a, r.b, r.c, r.d),
  pearson_chi_square: (r) => mod.pearsonChiSquare(r.table),
  mcnemar_classical: (r) => mod.mcnemarClassical(r.a, r.b, r.c, r.d),
  normal_quantile: (r) => mod.normalQuantile(r.p),
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
    // JSON.stringify serializes Infinity as null; preserve it explicitly.
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
