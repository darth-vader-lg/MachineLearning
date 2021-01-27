using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;

namespace MachineLearning.Data
{
   /// <summary>
   /// Vista di dati filtrata
   /// </summary>
   public sealed partial class DataViewFiltered : ChannelProvider, IDataAccess, IDataViewRowFilter
   {
      #region Fields
      /// <summary>
      /// Data view sorgente
      /// </summary>
      private readonly IDataAccess data;
      /// <summary>
      /// Filtro di righe
      /// </summary>
      private readonly DataViewRowFilter filter;
      #endregion
      #region Properties
      /// <summary>
      /// Indica se la dataview ha la capacita' di shuffling
      /// </summary>
      public bool CanShuffle => data.CanShuffle;
      /// <summary>
      /// Descrizione del contesto
      /// </summary>
      string IExceptionContext.ContextDescription => data.ContextDescription;
      /// <summary>
      /// Lo schema della dataview
      /// </summary>
      public DataViewSchema Schema => data.Schema;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="data">La sorgente di dati</param>
      /// <param name="filter">Eventuale filtro di riga esterno</param>
      private DataViewFiltered(IDataAccess data, DataViewRowFilter filter = null)
      {
         Contracts.AssertValue(data, nameof(data));
         data.AssertValueOrNull(filter);
         this.data = data;
         this.filter = filter ?? (column => true);
      }
      /// <summary>
      /// Crea una vista di dati filtrata
      /// </summary>
      /// <param name="data"></param>
      /// <param name="filter"></param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewFiltered Create(IDataAccess data, DataViewRowFilter filter = null)
      {
         Contracts.CheckValue(data, nameof(data));
         data.CheckValue(data, nameof(data));
         data.CheckValueOrNull(filter);
         return new DataViewFiltered(data, filter);
      }
      /// <summary>
      /// Funzione di ottenimento del provider di canali
      /// </summary>
      /// <returns>Il provider</returns>
      protected sealed override IChannelProvider GetChannelProvider() => data;
      /// <summary>
      /// Numero di righe
      /// </summary>
      /// <returns>Il numero di righe</returns>
      public long? GetRowCount() => data.GetRowCount();
      /// <summary>
      /// Restituisce un cursore
      /// </summary>
      /// <param name="columnsNeeded">Le colonne richieste</param>
      /// <param name="rand">Randomizzatore</param>
      /// <returns>Il cursore</returns>
      public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null) => new RowCursor(this, columnsNeeded, rand);
      /// <summary>
      /// Restituisce un set di cursori
      /// </summary>
      /// <param name="columnsNeeded">Le colonne richieste</param>
      /// <param name="n">Il grado di parallelismo suggerito</param>
      /// <param name="rand">Randomizzatore</param>
      /// <returns>Il cursore</returns>
      public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null) => new[] { new RowCursor(this, columnsNeeded, rand) };
      /// <summary>
      /// Validita' delle righe
      /// </summary>
      /// <param name="cursor">Cursore</param>
      /// <returns>La validita'</returns>
      bool IDataViewRowFilter.IsValidRow(DataViewRowCursor cursor) => filter(cursor);
      /// <summary>
      /// Avvia un canale standard di messaggistica
      /// </summary>
      /// <param name="name">Nome del canale</param>
      /// <returns>Il cancle</returns>
      IChannel IChannelProvider.Start(string name) => data.Start(name);
      /// <summary>
      /// Avvia un pipe standard di messaggistica
      /// </summary>
      /// <param name="name">Nome del canale</param>
      /// <returns>Il cancle</returns>
      IPipe<TMessage> IChannelProvider.StartPipe<TMessage>(string name) => data.StartPipe<TMessage>(name);
      /// <summary>
      /// Processo delle eccezioni
      /// </summary>
      /// <typeparam name="TException">Tipo di eccezione</typeparam>
      /// <param name="ex">Eccezione</param>
      /// <returns>L'eccezione</returns>
      TException IExceptionContext.Process<TException>(TException ex) => data.Process(ex);
      #endregion
   }

   public partial class DataViewFiltered // RowCursor
   {
      /// <summary>
      /// Classe per la gestione del cursore di riga fitrato
      /// </summary>
      private class RowCursor : DataViewRowCursor
      {
         #region Fields
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly DataViewFiltered owner;
         /// <summary>
         /// Cursore
         /// </summary>
         private readonly DataViewRowCursor cursor;
         #endregion
         #region Properties
         /// <summary>
         /// Posizione
         /// </summary>
         public override long Position => cursor.Position;
         /// <summary>
         /// Batch
         /// </summary>
         public override long Batch => cursor.Batch;
         /// <summary>
         /// Schema della IDataView
         /// </summary>
         public override DataViewSchema Schema => cursor.Schema;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="columnsNeeded">Colonne richieste</param>
         /// <param name="rand">Generatore di numeri casuali</param>
         public RowCursor(DataViewFiltered owner, IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
         {
            this.owner = owner;
            cursor = this.owner.data.GetRowCursor(columnsNeeded, rand);
         }
         /// <summary>
         /// Restituzione del Getter
         /// </summary>
         /// <typeparam name="TValue">Tipo del valore</typeparam>
         /// <param name="column">Colonna</param>
         /// <returns>Il getter</returns>
         public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) => cursor.GetGetter<TValue>(column);
         /// <summary>
         /// Restituzione del Getter di identificativo riga
         /// </summary>
         /// <returns>Il Getter dell'identificativo di riga</returns>
         public override ValueGetter<DataViewRowId> GetIdGetter() => cursor.GetIdGetter();
         /// <summary>
         /// Indicatore di colonna attiva
         /// </summary>
         /// <param name="column">La colonna da verificare</param>
         /// <returns>true se attiva</returns>
         public override bool IsColumnActive(DataViewSchema.Column column) => cursor.IsColumnActive(column);
         /// <summary>
         /// Funzione di avanzamento cursore
         /// </summary>
         /// <returns>true se il cursore e' puntato su una nuova riga</returns>
         public override bool MoveNext()
         {
            while (cursor.MoveNext()) {
               if ((owner as IDataViewRowFilter).IsValidRow(cursor))
                  return true;
            }
            return false;
         }
         #endregion
      }
   }
}
