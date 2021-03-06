using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe base per lo storage di dati di tipo binario
   /// </summary>
   [Serializable]
   public abstract class DataStorageBinary : DataStorageBase, IDataStorage
   {
      #region Properties
      /// <summary>
      /// Specifica se mantenere le colonne nascoste nel set di dati
      /// </summary>
      public bool KeepHidden { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Carica i dati in formato binario
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      protected IDataAccess LoadBinaryData(MLContext context)
      {
         Contracts.CheckValue(context, nameof(context));
         using (var checkHeader = GetReadStream()) {
            if (checkHeader == null)
               return null;
            checkHeader.Close();
         }
         return new DataAccess(context, context.Data.LoadFromBinary(this));
      }
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="textLoaderOptions">Opzioni di caricamento testuale (non utilizzate per i dati binari)</param>
      /// <returns>L'accesso ai dati</returns>
      public virtual IDataAccess LoadData(MLContext context, TextLoader.Options textLoaderOptions = default) => LoadBinaryData(context);
      /// <summary>
      /// Salva i dati in formato binario
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="stream">Stream per la scrittura. Lo stream viene chiuso automaticamente al termine della scrittura</param>
      protected void SaveBinaryData(MLContext context, IDataAccess data, Stream stream)
      {
         Contracts.CheckValue(context, nameof(context));
         lock (this) {
            try {
               if (stream == null)
                  throw new ArgumentException($"{nameof(stream)} cannot be null");
               if (!stream.CanWrite)
                  throw new ArgumentException($"{nameof(stream)} must be writable");
               // Salva
               context.Data.SaveAsBinary(data, stream, KeepHidden);
            }
            finally {
               try {
                  if (stream != null)
                     stream.Close();
               }
               catch (Exception) { }
            }
         }
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="textLoaderOptions">Opzioni di caricamento testuale (non utilizzate per i dati binari)</param>
      public abstract void SaveData(MLContext context, IDataAccess data, TextLoader.Options textLoaderOptions = default);
      #endregion
   }
}
