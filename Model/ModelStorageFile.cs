using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Model
{
   /// <summary>
   /// Gestore su file dello storage dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageFile : IModelStorage, IDataTimestamp
   {
      #region Properties
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp => File.GetLastWriteTimeUtc(FilePath);
      /// <summary>
      /// Path del file
      /// </summary>
      public string FilePath { get; private set; }
      /// <summary>
      /// Eventuale path di importazione di un modello esterno (ONNX / TensorFlow, ecc...)
      /// </summary>
      public string ImportPath { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file del modello</param>
      public ModelStorageFile(string filePath)
      {
         if (string.IsNullOrEmpty(filePath))
            throw new ArgumentException($"{nameof(filePath)} cannot be null");
         FilePath = filePath;
      }
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MLContext context, out DataViewSchema inputSchema)
      {
         Contracts.CheckValue(context, nameof(context));
         return context.Model.Load(FilePath, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(MLContext context, ITransformer model, DataViewSchema inputSchema)
      {
         Contracts.CheckValue(context, nameof(context));
         context.Model.Save(model, inputSchema, FilePath);
      }
      #endregion
   }
}
