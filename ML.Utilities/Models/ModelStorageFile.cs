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
      /// Storage di tipo stream
      /// </summary>
      private readonly IModelStorage modelStorage;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file del modello</param>
      public ModelStorageFile(string filePath) => modelStorage = new ModelStorageStream(() => File.OpenRead(filePath), () => File.OpenWrite(filePath));
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MLContext mlContext, out DataViewSchema schema) => modelStorage.LoadModel(mlContext, out schema);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="model">Modello da salvare</param>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      public void SaveModel(MLContext mlContext, ITransformer model, DataViewSchema schema) => modelStorage.SaveModel(mlContext, model, schema);
      #endregion
   }
}
