using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using MicrosoftExtensionsAiSample.Utils;
using OpenAI;
using System.ClientModel;

IEmbeddingGenerator<string, Embedding<float>>? embeddingGenerator
    = null;

// get the chat host
string embeddingGeneratorHost =
    ConsoleHelper.SelectFromOptions(
        [Statics.OllamaKey, Statics.OpenAiKey,
        Statics.AzureOpenAiKey]);

switch (embeddingGeneratorHost)
{
    // use OLLAMA
    case Statics.OllamaKey:

        // get the OLLAMA model name
        string ollamaModelName =
            ConsoleHelper.GetString("Enter your Ollama model name:");

        // create the OLLAMA embedding client
        embeddingGenerator = new OllamaEmbeddingGenerator(
            new Uri("http://localhost:11434/"), ollamaModelName);

        break;

    // use OpenAI
    case Statics.OpenAiKey:

        // get the OpenAI API key
        string openAiKey =
            ConsoleHelper.GetString("Enter your OpenAI API key:");

        // get the OpenAI model name
        string openAiModel =
            ConsoleHelper.SelectFromOptions(
                [Statics.TextEmbedding3SmallModelName,
                Statics.TextEmbedding3LargeModelName,
                Statics.TextEmbeddingAdaModelName]);

        // create the OpenAI embedding client
        embeddingGenerator = new OpenAIClient(
            openAiKey)
            .AsEmbeddingGenerator(openAiModel);

        break;

    // use Azure OpenAI
    case Statics.AzureOpenAiKey:

        // get the Azure OpenAI endpoint
        string azureOpenAiEndpoint =
            ConsoleHelper.GetString("Enter your Azure OpenAI endpoint:");

        // get the Azure OpenAI API key
        string azureOpenAiKey =
            ConsoleHelper.GetString("Enter your Azure OpenAI API key:");

        // get the Azure OpenAI model name
        string azureOpenAiModel =
            ConsoleHelper.GetString("Enter your Azure OpenAI embedding model name:");

        // create the Azure OpenAI embedding client
        embeddingGenerator = new AzureOpenAIClient(
            new Uri(azureOpenAiEndpoint),
            new ApiKeyCredential(azureOpenAiKey))
            .AsEmbeddingGenerator(azureOpenAiModel);

        break;
}

// check if the embedding client is valid
if (embeddingGenerator is null)
{
    ConsoleHelper.DisplayError("Invalid embedding host selected.");
    return;
}

// show the header
ConsoleHelper.ShowHeader();

// loop forever
while (true)
{
    // get the user message
    string embeddingInput =
        ConsoleHelper.GetString("Create embeddings for: ", false);

    Console.WriteLine();
    Console.WriteLine("Embeddings:");

    GeneratedEmbeddings<Embedding<float>> embedding =
        await embeddingGenerator.GenerateAsync([embeddingInput]);

    Console.WriteLine(string.Join(", ", embedding[0].Vector.ToArray()));

    Console.WriteLine();
    Console.WriteLine();
}