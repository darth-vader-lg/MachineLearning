using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;
using System.Text;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo testo in memoria
   /// </summary>
   [Serializable]
   public sealed class DataTextMemory : IDataStorage, IDataTextProvider, IMultiStreamSource, ITimestamp
   {
      #region Fields
      /// <summary>
      /// Dati testuali
      /// </summary>
      private string _textData;
      #endregion
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => 1;
      /// <summary>
      /// Dati testuali
      /// </summary>
      public string TextData { get => _textData; set { _textData = value; Timestamp = DateTime.UtcNow; } }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime Timestamp { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public DataTextMemory() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">Dati</param>
      public DataTextMemory(object context, IDataView data) => SaveData(context, data);
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">Dati</param>
      public DataTextMemory(object context, string data)
      {
         TextData = data;
         SaveData(context, LoadData(context));
      }
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Il path</returns>
      string IMultiStreamSource.GetPathOrNull(int index) => null;
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => new MemoryStream(Encoding.Default.GetBytes(TextData ?? ""));
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => new StringReader(TextData ?? "");
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(object context)
      {
         var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
         var opt = (context as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? new TextLoader.Options();
         return ml.Data.CreateTextLoader(opt).Load(this);
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      public void SaveData(object context, IDataView data, bool schema = false)
      {
         lock (this) {
            // Data e ora
            var timestamp = DateTime.UtcNow;
            // Oggetto per la scrittura dei dati in memoria
            using var writer = new MemoryStream();
            // Contesto e opzioni
            var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
            var opt = (context as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? new TextLoader.Options();
            // Separatore di colonne
            var separator = opt.Separators?.FirstOrDefault() ?? '\t';
            separator = separator != default ? separator : '\t';
            // Salva come testo i dati
            ml.Data.SaveAsText(
               data: data,
               stream: writer,
               separatorChar: separator,
               headerRow: opt.HasHeader,
               schema: schema,
               keepHidden: true,
               forceDense: false);
            // Crea uno stream per la lettura
            writer.Position = 0;
            using var reader = new StreamReader(writer);
            // Aggiorna la stringa
            TextData = reader.ReadToEnd();
            Timestamp = timestamp;
         }
      }
      #endregion
   }
}
