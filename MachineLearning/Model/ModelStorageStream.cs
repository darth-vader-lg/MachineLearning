using MachineLearning.Data;
using System;
using System.Collections.Generic;
using System.IO;

namespace MachineLearning.Model
{
   /// <summary>
   /// Gestore dello storage su stream dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageStream : IModelStorage, IDataTimestamp
   {
      #region Properties
      /// <summary>
      /// Funzione di ottenimento dello stream di lettura
      /// </summary>
      [field: NonSerialized]
      public CompositeModel.StreamGetter StreamGetter { get; set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp { get; private set; } = DateTime.UtcNow;
      #endregion
      #region Methods
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
               var stream = StreamGetter?.Invoke(index, write);
               if (stream == null)
                  return null;
               streams.Add(stream);
               return streams[streams.Count - 1];
            });
            return model;
         }
         finally {
            streams.ForEach(s => s.Close());
         }
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <typeparam name="T">Il tipo di contesto</typeparam>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      public void SaveModel(IMachineLearningContextProvider context, CompositeModel model)
      {
         var streams = new List<Stream>();
         try {
            model.Save((index, write) =>
            {
               var stream = StreamGetter?.Invoke(index, write);
               if (stream == null)
                  throw new InvalidOperationException("Cannot write to the stream");
               streams.Add(stream);
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
