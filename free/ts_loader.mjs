// Resolve-hook that appends a .ts extension when an extensionless specifier
// maps to a local .ts file. Lets the Zetyra frontend's `import "./chi_square"`
// idiom work under Node's native strip-types loader the same way it works
// under tsc/Next.js bundler resolution.

import { access, constants } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve as pathResolve } from "node:path";

export async function resolve(specifier, context, nextResolve) {
  if ((specifier.startsWith("./") || specifier.startsWith("../")) &&
      !/\.[a-zA-Z]+$/.test(specifier) &&
      context.parentURL?.startsWith("file://")) {
    const parentDir = dirname(fileURLToPath(context.parentURL));
    const candidate = pathResolve(parentDir, specifier + ".ts");
    try {
      await access(candidate, constants.R_OK);
      return nextResolve(pathToFileURL(candidate).href, context);
    } catch {
      // fall through to default resolution
    }
  }
  return nextResolve(specifier, context);
}
