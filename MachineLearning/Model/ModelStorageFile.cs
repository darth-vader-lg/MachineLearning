using MachineLearning.Data;
using System;
using System.Collections.Generic;
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
      /// Path della directory contenente il modello
      /// </summary>
      public string ModelDir { get; private set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp => Directory.GetLastWriteTimeUtc(ModelDir);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="modelDir">Path della directory del modello</param>
      public ModelStorageFile(string modelDir)
      {
         if (string.IsNullOrEmpty(modelDir))
            throw new ArgumentException($"{nameof(modelDir)} cannot be null");
         ModelDir = modelDir;
      }
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>Il modello</returns>
      public CompositeModel LoadModel(IMachineLearningContextProvider context)
      {
         var streams = new List<Stream>();
         try {
            var model = new CompositeModel(context, (index, write) =>
            {
               var file = Path.Combine(ModelDir, index.ToString("000"));
               if (File.Exists(file)) {
                  streams.Add(File.OpenRead(file));
                  return streams[streams.Count - 1];
               }
               return null;
            });
            return model.Load();
         }
         finally {
            streams.ForEach(s => s.Close());
         }
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      public void SaveModel(IMachineLearningContextProvider context, CompositeModel model)
      {
         if (!Directory.Exists(ModelDir))
            Directory.CreateDirectory(ModelDir);
         var streams = new List<Stream>();
         try {
            model.Save((index, write) =>
            {
               var file = Path.Combine(ModelDir, index.ToString("000"));
               streams.Add(File.Create(file));
               return streams[streams.Count - 1];
            });
         }
         finally {
            streams.ForEach(s => s.Close());
         }
      }
      #endregion
   }
}
