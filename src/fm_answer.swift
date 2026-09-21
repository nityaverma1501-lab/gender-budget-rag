// Thin CLI bridge to Apple's on-device Foundation Model (Apple Intelligence).
// Reads a JSON object {"instructions": "...", "prompt": "..."} from stdin,
// prints the model's raw text response to stdout.
import FoundationModels
import Foundation

let inputData = FileHandle.standardInput.readDataToEndOfFile()
guard let obj = try? JSONSerialization.jsonObject(with: inputData) as? [String: String],
      let instructions = obj["instructions"], let prompt = obj["prompt"] else {
    FileHandle.standardError.write("Invalid input JSON, expected {instructions, prompt}\n".data(using: .utf8)!)
    exit(1)
}

let session = LanguageModelSession(instructions: instructions)
let options = GenerationOptions(temperature: 0.0)
do {
    let response = try await session.respond(to: prompt, options: options)
    print(response.content)
} catch {
    FileHandle.standardError.write("ERROR: \(error)\n".data(using: .utf8)!)
    exit(1)
}
