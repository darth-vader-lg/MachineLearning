using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;

namespace MachineLearning
{
   /// <summary>
   /// Vista di dati filtrata
   /// </summary>
   public sealed partial class DataViewFiltered : IDataView, IDataViewRowFilter
   {
      #region Fields
      /// <summary>
      /// Data view sorgente
      /// </summary>
      private readonly IDataView _dataView;
      /// <summary>
      /// Filtro di righe
      /// </summary>
      private readonly DataViewRowFilter _filter;
      /// <summary>
      /// Host
      /// </summary>
      private readonly IHost _host;
      #endregion
      #region Properties
      /// <summary>
      /// Indica se la dataview ha la capacita' di shuffling
      /// </summary>
      public bool CanShuffle => _dataView.CanShuffle;
      /// <summary>
      /// Lo schema della dataview
      /// </summary>
      public DataViewSchema Schema => _dataView.Schema;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine earning</param>
      /// <param name="dataView">La sorgente di dati</param>
      /// <param name="filter">Eventuale filtro di riga esterno</param>
      private DataViewFiltered(IMachineLearningContextProvider context, IDataView dataView, DataViewRowFilter filter = null)
      {
         Contracts.CheckValue(context?.ML?.NET, nameof(context));
         _host = (context.ML.NET as IHostEnvironment).Register(nameof(DataViewFiltered));
         _host.AssertValue(dataView, nameof(dataView));
         _host.AssertValueOrNull(filter);
         _dataView = dataView;
         _filter = filter ?? (this as IDataViewRowFilter).IsValidRow;
      }
      /// <summary>
      /// Crea una vista di dati filtrata
      /// </summary>
      /// <param name="context"></param>
      /// <param name="dataView"></param>
      /// <param name="filter"></param>
      /// <returns>La vista di dati filtrata</returns>
      public static DataViewFiltered Create(IMachineLearningContextProvider context, IDataView dataView, DataViewRowFilter filter = null)
      {
         Contracts.CheckValue(context?.ML?.NET, nameof(context));
         context.ML.NET.CheckValue(dataView, nameof(dataView));
         context.ML.NET.CheckValueOrNull(filter);
         return new DataViewFiltered(context, dataView, filter);
      }
      /// <summary>
      /// Numero di righe
      /// </summary>
      /// <returns>Il numero di righe</returns>
      public long? GetRowCount() => _dataView.GetRowCount();
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
      /// <param name="row">Riga</param>
      /// <returns>Sempre true</returns>
      bool IDataViewRowFilter.IsValidRow(DataViewRow row) => true;
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
         private readonly DataViewFiltered _owner;
         /// <summary>
         /// Cursore
         /// </summary>
         private readonly DataViewRowCursor _cursor;
         #endregion
         #region Properties
         /// <summary>
         /// Posizione
         /// </summary>
         public override long Position => _cursor.Position;
         /// <summary>
         /// Batch
         /// </summary>
         public override long Batch => _cursor.Batch;
         /// <summary>
         /// Schema della IDataView
         /// </summary>
         public override DataViewSchema Schema => _cursor.Schema;
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
            _owner = owner;
            _cursor = _owner._dataView.GetRowCursor(columnsNeeded, rand);
         }
         /// <summary>
         /// Restituzione del Getter
         /// </summary>
         /// <typeparam name="TValue">Tipo del valore</typeparam>
         /// <param name="column">Colonna</param>
         /// <returns>Il getter</returns>
         public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) => _cursor.GetGetter<TValue>(column);
         /// <summary>
         /// Restituzione del Getter di identificativo riga
         /// </summary>
         /// <returns>Il Getter dell'identificativo di riga</returns>
         public override ValueGetter<DataViewRowId> GetIdGetter() => _cursor.GetIdGetter();
         /// <summary>
         /// Indicatore di colonna attiva
         /// </summary>
         /// <param name="column">La colonna da verificare</param>
         /// <returns>true se attiva</returns>
         public override bool IsColumnActive(DataViewSchema.Column column) => _cursor.IsColumnActive(column);
         /// <summary>
         /// Funzione di avanzamento cursore
         /// </summary>
         /// <returns>true se il cursore e' puntato su una nuova riga</returns>
         public override bool MoveNext()
         {
            while (_cursor.MoveNext() && !_owner._filter(_cursor)) ;
            return false;
         }
         #endregion
      }
   }
}
