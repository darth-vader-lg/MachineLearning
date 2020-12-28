using Microsoft.ML;
using System;
using System.IO;

namespace MachineLearning
{
   /// <summary>
   /// Gestore su file dello storage dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageFile : IModelStorage, ITimestamp
   {
      #region Fields
      /// <summary>
      /// Storage di tipo stream
      /// </summary>
      [NonSerialized]
      private IModelStorage _modelStorage;
      #endregion
      #region Properties
      /// <summary>
      /// Path del file
      /// </summary>
      public string FilePath { get; private set; }
      /// <summary>
      /// Storage di tipo stream
      /// </summary>
      private IModelStorage Storage => _modelStorage ??= new ModelStorageStream(() => File.OpenRead(FilePath), () => File.Create(FilePath));
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime Timestamp => File.GetLastWriteTimeUtc(FilePath);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file del modello</param>
      public ModelStorageFile(string filePath) => FilePath = filePath;
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MachineLearningContext ml, out DataViewSchema inputSchema)
      {
         if (string.IsNullOrEmpty(FilePath) || !File.Exists(FilePath))
            throw new FileNotFoundException("File not found", FilePath);
         return Storage.LoadModel(ml, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(MachineLearningContext ml, ITransformer model, DataViewSchema inputSchema) => Storage.SaveModel(ml, model, inputSchema);
      #endregion
   }
}
