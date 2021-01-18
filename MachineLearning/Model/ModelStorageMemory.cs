using MachineLearning.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MachineLearning.Model
{
   /// <summary>
   /// Gestore dello storage in memoria dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageMemory : IModelStorage, IDataTimestamp
   {
      #region Properties
      /// <summary>
      /// Bytes del modello
      /// </summary>
      public IEnumerable<byte[]> Bytes { get; private set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp { get; private set; } = DateTime.UtcNow;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public ModelStorageMemory() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="memory">Contenuto iniziale</param>
      public ModelStorageMemory(IEnumerable<byte[]> bytes) => Bytes = bytes;
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>Il modello</returns>
      public CompositeModel LoadModel(IMachineLearningContextProvider context)
      {
         if (Bytes == default)
            return default;
         var enumerator = Bytes.GetEnumerator();
         var model = new CompositeModel(context, (index, write) =>
         {
            if (!enumerator.MoveNext())
               return null;
            return new MemoryStream(enumerator.Current);
         });
         return model;
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      public void SaveModel(IMachineLearningContextProvider context, CompositeModel model)
      {
         lock (this) {
            var timestamp = DateTime.UtcNow;
            var bytes = new MemoryStream[model.Count];
            model.Save((index, write) => bytes[index] = new MemoryStream());
            Bytes = from b in bytes select b.ToArray();
            DataTimestamp = timestamp;
         }
      }
      #endregion
   }
}
