using Microsoft.ML;
using System;
using System.IO;

namespace ML.Utilities.Models
{
   /// <summary>
   /// Gestore su file dello storage dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageFile : IModelStorage
   {
      #region Fields
      /// <summary>
      /// Path del file
      /// </summary>
      private readonly string _filePath;
      /// <summary>
      /// Storage di tipo stream
      /// </summary>
      [NonSerialized]
      private IModelStorage _modelStorage;
      #endregion
      #region Properties
      /// <summary>
      /// Storage di tipo stream
      /// </summary>
      private IModelStorage Storage => _modelStorage ??= new ModelStorageStream(() => File.OpenRead(_filePath), () => File.OpenWrite(_filePath));
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file del modello</param>
      public ModelStorageFile(string filePath) => _filePath = filePath;
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MachineLearningContext ml, out DataViewSchema inputSchema) => Storage.LoadModel(ml, out inputSchema);
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
