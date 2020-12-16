using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace ML.Utilities.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo stringhe
   /// </summary>
   [Serializable]
   public sealed partial class DataStorageString : IDataStorage, IDataTextProvider
   {
      #region Properties
      /// <summary>
      /// Configurazione dei dati
      /// </summary>
      public TextLoader.Options Config { get; set; } = new TextLoader.Options();
      /// <summary>
      /// Dati testuali
      /// </summary>
      public string TextData { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(MLContext mlContext) => mlContext.Data.CreateTextLoader(Config ?? new TextLoader.Options()).Load(new Source(this));
      #endregion
   }

   /// <summary>
   /// La sorgente dei dati
   /// </summary>
   public partial class DataStorageString
   {
      private class Source : IMultiStreamSource
      {
         #region Fields
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly IDataTextProvider owner;
         #endregion
         #region Properties
         /// <summary>
         /// Il numero di items
         /// </summary>
         public int Count => 1;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         public Source(IDataTextProvider owner) => this.owner = owner;
         /// <summary>
         /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Sempre null</returns>
         public string GetPathOrNull(int index) => default;
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public Stream Open(int index)
         {
            var memoryStream = new MemoryStream();
            using var writer = new StreamWriter(memoryStream);
            writer.Write(owner.TextData ?? "");
            memoryStream.Position = 0;
            return memoryStream;
         }
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public TextReader OpenTextReader(int index) => new StringReader(owner.TextData ?? "");
         #endregion
      }
   }
}
