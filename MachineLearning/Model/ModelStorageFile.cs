using MachineLearning.Data;
using Microsoft.ML;
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
      /// Path del file
      /// </summary>
      public string FilePath { get; private set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp => File.GetLastWriteTimeUtc(FilePath);
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
      /// <typeparam name="T">Il tipo di contesto</typeparam>
      /// <param name="context">Contesto</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(IMachineLearningContextProvider context, out DataViewSchema inputSchema) =>
         (context?.ML?.NET ?? new MLContext()).Model.Load(FilePath, out inputSchema);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <typeparam name="T">Il tipo di contesto</typeparam>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(IMachineLearningContextProvider context, ITransformer model, DataViewSchema inputSchema) =>
         (context?.ML?.NET ?? new MLContext()).Model.Save(model, inputSchema, FilePath);
      #endregion
   }
}
